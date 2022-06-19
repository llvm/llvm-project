; RUN: opt -mtriple=x86_64-linux-gnu -dwarfehprepare -simplifycfg-require-and-preserve-domtree=1 -run-twice < %s -S | FileCheck %s

; Check basic functionality of IR-to-IR DWARF EH preparation. This should
; eliminate resumes. This pass requires a TargetMachine, so we put it under X86
; and provide an x86 triple.

@int_typeinfo = global i8 0

declare void @might_throw()
declare void @cleanup()

define i32 @simple_cleanup_catch() personality i32 (...)* @__gxx_personality_v0 {
  invoke void @might_throw()
          to label %cont unwind label %lpad

; CHECK-LABEL: define i32 @simple_cleanup_catch()
; CHECK: invoke void @might_throw()

cont:
  ret i32 0

; CHECK: ret i32 0

lpad:
  %ehvals = landingpad { i8*, i32 }
      cleanup
      catch i8* @int_typeinfo
  %ehptr = extractvalue { i8*, i32 } %ehvals, 0
  %ehsel = extractvalue { i8*, i32 } %ehvals, 1
  call void @cleanup()
  %int_sel = call i32 @llvm.eh.typeid.for(i8* @int_typeinfo)
  %int_match = icmp eq i32 %ehsel, %int_sel
  br i1 %int_match, label %catch_int, label %eh.resume

; CHECK: lpad:
; CHECK: landingpad { i8*, i32 }
; CHECK: call void @cleanup()
; CHECK: call i32 @llvm.eh.typeid.for
; CHECK: br i1

catch_int:
  ret i32 1

; CHECK: catch_int:
; CHECK: ret i32 1

eh.resume:
  %tmp_ehvals = insertvalue { i8*, i32 } undef, i8* %ehptr, 0
  %new_ehvals = insertvalue { i8*, i32 } %tmp_ehvals, i32 %ehsel, 1
  resume { i8*, i32 } %new_ehvals

; CHECK: eh.resume:
; CHECK-NEXT: call void @_Unwind_Resume(i8* %ehptr)
}


define i32 @catch_no_resume() personality i32 (...)* @__gxx_personality_v0 {
  invoke void @might_throw()
          to label %cont unwind label %lpad

cont:
  ret i32 0

lpad:
  %ehvals = landingpad { i8*, i32 }
      catch i8* @int_typeinfo
  %ehptr = extractvalue { i8*, i32 } %ehvals, 0
  %ehsel = extractvalue { i8*, i32 } %ehvals, 1
  %int_sel = call i32 @llvm.eh.typeid.for(i8* @int_typeinfo)
  %int_match = icmp eq i32 %ehsel, %int_sel
  br i1 %int_match, label %catch_int, label %eh.resume

catch_int:
  ret i32 1

eh.resume:
  %tmp_ehvals = insertvalue { i8*, i32 } undef, i8* %ehptr, 0
  %new_ehvals = insertvalue { i8*, i32 } %tmp_ehvals, i32 %ehsel, 1
  resume { i8*, i32 } %new_ehvals
}

; Check that we can prune the unreachable resume instruction.

; CHECK-LABEL: define i32 @catch_no_resume() personality i32 (...)* @__gxx_personality_v0 {
; CHECK: invoke void @might_throw()
; CHECK: ret i32 0
; CHECK: lpad:
; CHECK: landingpad { i8*, i32 }
; CHECK-NOT: br i1
; CHECK: ret i32 1
; CHECK-NOT: call void @_Unwind_Resume
; CHECK: {{^[}]}}


define i32 @catch_cleanup_merge() personality i32 (...)* @__gxx_personality_v0 {
  invoke void @might_throw()
          to label %inner_invoke unwind label %outer_lpad
inner_invoke:
  invoke void @might_throw()
          to label %cont unwind label %inner_lpad
cont:
  ret i32 0

outer_lpad:
  %ehvals1 = landingpad { i8*, i32 }
      catch i8* @int_typeinfo
  br label %catch.dispatch

inner_lpad:
  %ehvals2 = landingpad { i8*, i32 }
      cleanup
      catch i8* @int_typeinfo
  call void @cleanup()
  br label %catch.dispatch

catch.dispatch:
  %ehvals = phi { i8*, i32 } [ %ehvals1, %outer_lpad ], [ %ehvals2, %inner_lpad ]
  %ehptr = extractvalue { i8*, i32 } %ehvals, 0
  %ehsel = extractvalue { i8*, i32 } %ehvals, 1
  %int_sel = call i32 @llvm.eh.typeid.for(i8* @int_typeinfo)
  %int_match = icmp eq i32 %ehsel, %int_sel
  br i1 %int_match, label %catch_int, label %eh.resume

catch_int:
  ret i32 1

eh.resume:
  %tmp_ehvals = insertvalue { i8*, i32 } undef, i8* %ehptr, 0
  %new_ehvals = insertvalue { i8*, i32 } %tmp_ehvals, i32 %ehsel, 1
  resume { i8*, i32 } %new_ehvals
}

; We can't prune this merge because one landingpad is a cleanup pad.

; CHECK-LABEL: define i32 @catch_cleanup_merge()
; CHECK: invoke void @might_throw()
; CHECK: invoke void @might_throw()
; CHECK: ret i32 0
;
; CHECK: outer_lpad:
; CHECK: landingpad { i8*, i32 }
; CHECK: br label %catch.dispatch
;
; CHECK: inner_lpad:
; CHECK: landingpad { i8*, i32 }
; CHECK: call void @cleanup()
; CHECK: br label %catch.dispatch
;
; CHECK: catch.dispatch:
; CHECK: call i32 @llvm.eh.typeid.for
; CHECK: br i1
; CHECK: catch_int:
; CHECK: ret i32 1
; CHECK: eh.resume:
; CHECK-NEXT: call void @_Unwind_Resume(i8* %ehptr)

declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)
