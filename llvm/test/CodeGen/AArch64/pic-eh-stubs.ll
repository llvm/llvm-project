; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -simplifycfg-require-and-preserve-domtree=1 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64_be-none-linux-gnu -relocation-model=pic -simplifycfg-require-and-preserve-domtree=1 -o - %s | FileCheck %s

; Make sure exception-handling PIC code can be linked correctly. An alternative
; to the sequence described below would have .gcc_except_table itself writable
; and not use the indirection, but this isn't what LLVM does right now.

  ; There should be a read-only .gcc_except_table section...
; CHECK: .section .gcc_except_table,"a"

  ; ... referring indirectly to stubs for its typeinfo ...
; CHECK: // @TType Encoding = indirect pcrel sdata8
  ; ... one of which is "int"'s typeinfo
; CHECK: [[TYPEINFO_LBL:.Ltmp[0-9]+]]: // TypeInfo 1
; CHECK-NEXT: .xword  .L_ZTIi.DW.stub-[[TYPEINFO_LBL]]

  ; .. and which is properly defined (in a writable section for the dynamic loader) later.
; CHECK: .data
; CHECK: .L_ZTIi.DW.stub:
; CHECK-NEXT: .xword _ZTIi

@_ZTIi = external constant ptr

define i32 @_Z3barv() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z3foov()
          to label %return unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi) nounwind
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3) nounwind
  %exn.scalar = load i32, ptr %4, align 4
  tail call void @__cxa_end_catch() nounwind
  br label %return

return:                                           ; preds = %entry, %catch
  %retval.0 = phi i32 [ %exn.scalar, %catch ], [ 42, %entry ]
  ret i32 %retval.0

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } %0
}

declare void @_Z3foov()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()
