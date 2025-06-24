; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -win-eh-prepare < %s | FileCheck %s

; This test verifies the fix for the SEH double-finally bug where calls after
; finally blocks could cause the finally block to execute twice during unwinding.

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Test the exact double-finally bug pattern
define void @test_double_finally_fix() personality ptr @__C_specific_handler {
entry:
  %context = alloca ptr, align 8
  %status = alloca i32, align 4
  store ptr null, ptr %context, align 8
  store i32 0, ptr %status, align 4
  
  ; SEH try block with potential exception
  invoke void @risky_operation(ptr %context)
          to label %try_success unwind label %exception_handler

try_success:
  ; Normal completion - do finally cleanup
  call void @cleanup_resources(ptr %context)
  br label %after_finally

exception_handler:
  ; Exception path - also do finally cleanup
  %lpad = landingpad { ptr, i32 }
           cleanup
  call void @cleanup_resources(ptr %context)
  br label %after_finally_from_exception

after_finally:
  ; CRITICAL: This call after finally could cause double execution
  ; The fix should convert this call to invoke to prevent the bug
  ; CHECK: invoke void @error_reporting_function(ptr %context, i32 -1073741773)
  ; CHECK-NEXT: to label %{{[a-zA-Z0-9_.]+}} unwind label %{{[a-zA-Z0-9_.]+}}
  ; CHECK-NOT: call void @error_reporting_function
  call void @error_reporting_function(ptr %context, i32 -1073741773)
  
  ret void

after_finally_from_exception:
  ; Also test the exception path
  ; CHECK: invoke void @error_reporting_function(ptr %context, i32 -1073741773)
  ; CHECK-NEXT: to label %{{[a-zA-Z0-9_.]+}} unwind label %{{[a-zA-Z0-9_.]+}}
  call void @error_reporting_function(ptr %context, i32 -1073741773)
  resume { ptr, i32 } %lpad

; Our fix should create proper unwind destinations with landing pads
; CHECK: %{{[a-zA-Z0-9_.]+}} = landingpad { ptr, i32 }
; CHECK-NEXT: cleanup
; CHECK: resume { ptr, i32 } %{{[a-zA-Z0-9_.]+}}
}

; Test that the fix works with functions containing "cleanup" to trigger heuristic
define void @test_cleanup_function_trigger() personality ptr @__C_specific_handler {
entry:
  %ptr = alloca ptr, align 8
  
  ; This call should trigger our heuristic detection
  call void @do_some_cleanup_work(ptr %ptr)
  
  ; This call should be converted because function has cleanup operations
  ; CHECK: invoke void @potentially_throwing_function(ptr %ptr, i32 1)
  ; CHECK-NEXT: to label %{{[a-zA-Z0-9_.]+}} unwind label %{{[a-zA-Z0-9_.]+}}
  call void @potentially_throwing_function(ptr %ptr, i32 1)
  
  ret void
}

; Simpler test with just a cleanup landing pad to trigger our detection
define void @test_simple_cleanup_trigger() personality ptr @__C_specific_handler {
entry:
  %ptr = alloca ptr, align 8
  
  ; Create SEH context with cleanup
  invoke void @work_function(ptr %ptr)
          to label %normal unwind label %cleanup

normal:
  ; This call should be converted to invoke
  ; CHECK: invoke void @potentially_throwing_function(ptr %ptr, i32 2)
  ; CHECK-NEXT: to label %{{[a-zA-Z0-9_.]+}} unwind label %{{[a-zA-Z0-9_.]+}}
  call void @potentially_throwing_function(ptr %ptr, i32 2)
  ret void

cleanup:
  %lpad = landingpad { ptr, i32 }
           cleanup
  resume { ptr, i32 } %lpad
}

; Test that nounwind calls are NOT converted (should remain safe)
define void @test_nounwind_not_converted() personality ptr @__C_specific_handler {
entry:
  %ptr = alloca ptr, align 8
  
  ; Create SEH context
  invoke void @work_function(ptr %ptr)
          to label %normal unwind label %cleanup

normal:
  ; Nounwind calls should remain as calls even in SEH context
  ; CHECK: call void @safe_nounwind_function(ptr %ptr)
  ; CHECK-NOT: invoke void @safe_nounwind_function
  call void @safe_nounwind_function(ptr %ptr) nounwind
  ret void

cleanup:
  %lpad = landingpad { ptr, i32 }
           cleanup
  resume { ptr, i32 } %lpad
}

; Test that non-SEH functions are unaffected  
define void @test_non_seh_unaffected() {
entry:
  %ptr = alloca ptr, align 8
  
  call void @cleanup_resources(ptr %ptr)
  
  ; Without SEH personality, calls should remain unchanged
  ; CHECK-LABEL: @test_non_seh_unaffected
  ; CHECK: call void @potentially_throwing_function(ptr %ptr, i32 3)
  ; CHECK-NOT: invoke void @potentially_throwing_function{{.*}}i32 3
  call void @potentially_throwing_function(ptr %ptr, i32 3)
  
  ret void
}

; Test the x86 32-bit SEH personality as well
define void @test_x86_seh() personality ptr @_except_handler3 {
entry:
  %ptr = alloca ptr, align 4
  
  ; Create context with cleanup
  invoke void @work_function(ptr %ptr)
          to label %normal unwind label %cleanup

normal:
  ; Should work with 32-bit SEH too
  ; CHECK: invoke void @potentially_throwing_function(ptr %ptr, i32 4)
  ; CHECK-NEXT: to label %{{[a-zA-Z0-9_.]+}} unwind label %{{[a-zA-Z0-9_.]+}}
  call void @potentially_throwing_function(ptr %ptr, i32 4)
  ret void

cleanup:
  %lpad = landingpad { ptr, i32 }
           cleanup
  resume { ptr, i32 } %lpad
}

; Test with intrinsics that should NOT be converted
define void @test_intrinsics_not_converted() personality ptr @__C_specific_handler {
entry:
  %ptr = alloca ptr, align 8
  
  invoke void @work_function(ptr %ptr)
          to label %normal unwind label %cleanup

normal:
  ; LLVM intrinsics should not be converted
  ; CHECK: call void @llvm.memset.p0.i64(ptr %ptr, i8 0, i64 8, i1 false)
  ; CHECK-NOT: invoke void @llvm.memset
  call void @llvm.memset.p0.i64(ptr %ptr, i8 0, i64 8, i1 false)
  ret void

cleanup:
  %lpad = landingpad { ptr, i32 }
           cleanup
  resume { ptr, i32 } %lpad
}

declare void @__C_specific_handler(...)
declare void @_except_handler3(...)
declare void @risky_operation(ptr)
declare void @cleanup_resources(ptr)
declare void @error_reporting_function(ptr, i32)
declare void @do_some_cleanup_work(ptr)
declare void @potentially_throwing_function(ptr, i32)
declare void @safe_nounwind_function(ptr) nounwind
declare void @work_function(ptr)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) nounwind