; RUN: llc < %s -mtriple=i686-pc-linux-gnu -o - | FileCheck %s

; CHECK: .cfi_personality 0, __gnat_eh_personality
; CHECK: .cfi_lsda 0, .Lexception0

@error = external global i8

define void @_ada_x() {
entry:
  invoke void @raise()
          to label %eh_then unwind label %unwind

unwind:                                           ; preds = %entry 
  %eh_ptr = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*)
              catch i8* @error
  %eh_select = extractvalue { i8*, i32 } %eh_ptr, 1
  %eh_typeid = tail call i32 @llvm.eh.typeid.for(i8* @error)
  %tmp2 = icmp eq i32 %eh_select, %eh_typeid
  br i1 %tmp2, label %eh_then, label %Unwind

eh_then:                                          ; preds = %unwind, %entry
  ret void

Unwind:                                           ; preds = %unwind
  resume { i8*, i32 } %eh_ptr
}

declare void @raise()

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)
