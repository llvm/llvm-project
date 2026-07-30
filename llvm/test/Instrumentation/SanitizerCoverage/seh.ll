; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=0 -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -S | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

declare i32 @llvm.eh.typeid.for(ptr) #2
declare ptr @llvm.frameaddress(i32)
declare ptr @llvm.eh.recoverfp(ptr, ptr)
declare ptr @llvm.localrecover(ptr, ptr, i32)
declare void @llvm.localescape(...) #1

declare i32 @_except_handler3(...)
declare void @may_throw(ptr %r)

define i32 @main() sanitize_address personality ptr @_except_handler3 {
entry:
  %r = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr nonnull %__exception_code)
  store i32 0, ptr %r, align 4
  invoke void @may_throw(ptr nonnull %r) #4
          to label %__try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @"\01?filt$0@0@main@@"
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = call i32 @llvm.eh.typeid.for(ptr @"\01?filt$0@0@main@@") #1
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %__except, label %eh.resume

__except:                                         ; preds = %lpad
  store i32 1, ptr %r, align 4
  br label %__try.cont

__try.cont:                                       ; preds = %entry, %__except
  %3 = load i32, ptr %r, align 4
  ret i32 %3

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } %0
}

; Check that we don't do any instrumentation.

; CHECK-LABEL: define i32 @main()
; CHECK-NOT: load atomic i32, ptr {{.*}} monotonic, align 4, !nosanitize
; CHECK-NOT: call void @__sanitizer_cov
; CHECK: ret i32

; Function Attrs: nounwind
define internal i32 @"\01?filt$0@0@main@@"() #1 {
entry:
  %0 = tail call ptr @llvm.frameaddress(i32 1)
  %1 = tail call ptr @llvm.eh.recoverfp(ptr @main, ptr %0)
  %2 = tail call ptr @llvm.localrecover(ptr @main, ptr %1, i32 0)
  %3 = getelementptr inbounds i8, ptr %0, i32 -20
  %4 = load ptr, ptr %3, align 4
  %5 = getelementptr inbounds { ptr, ptr }, ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 4
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  ret i32 1
}

; CHECK-LABEL: define internal i32 @"\01?filt$0@0@main@@"()
; CHECK: tail call ptr @llvm.localrecover(ptr @main, ptr {{.*}}, i32 0)

define void @ScaleFilterCols_SSSE3(ptr %dst_ptr, ptr %src_ptr, i32 %dst_width, i32 %x, i32 %dx) sanitize_address {
entry:
  %dst_width.addr = alloca i32, align 4
  store i32 %dst_width, ptr %dst_width.addr, align 4
  %0 = call { ptr, ptr, i32, i32, i32 } asm sideeffect "", "=r,=r,={ax},=r,=r,=*rm,rm,rm,0,1,2,3,4,5,~{memory},~{cc},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) nonnull %dst_width.addr, i32 %x, i32 %dx, ptr %dst_ptr, ptr %src_ptr, i32 0, i32 0, i32 0, i32 %dst_width)
  ret void
}

define void @ScaleColsUp2_SSE2() sanitize_address {
entry:
  ret void
}
