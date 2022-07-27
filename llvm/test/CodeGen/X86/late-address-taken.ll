; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s -enable-shrink-wrap=false | FileCheck %s
; Make sure shrink-wrapping does not break the lowering of exception handling.
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s -enable-shrink-wrap=true -pass-remarks-output=%t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=REMARKS

; Repro cases from PR25168

; test @catchret - catchret target is not address-taken until PEI
; splits it into lea/mov followed by ret.  Make sure the MBB is
; handled, both by tempting BranchFolding to merge it with %early_out
; and delete it, and by checking that we emit a proper reference
; to it in the LEA

declare void @ProcessCLRException()
declare void @f()

define void @catchret(i1 %b) personality ptr @ProcessCLRException {
entry:
  br i1 %b, label %body, label %early_out
early_out:
  ret void
body:
  invoke void @f()
          to label %exit unwind label %catch.pad
catch.pad:
  %cs1 = catchswitch within none [label %catch.body] unwind to caller
catch.body:
  %catch = catchpad within %cs1 [i32 33554467]
  catchret from %catch to label %exit
exit:
  ret void
}
; CHECK-LABEL: catchret:  # @catchret
; CHECK: [[Exit:^[^ :]+]]: # Block address taken
; CHECK-NEXT:              # %exit
; CHECK: # %catch.body
; CHECK: .seh_endprolog
; CHECK: leaq [[Exit]](%rip), %rax
; CHECK: retq # CATCHRET

; REMARKS: Pass:            shrink-wrap
; REMARKS-NEXT: Name:            UnsupportedEHFunclets
; REMARKS-NEXT: Function:        catchret
; REMARKS-NEXT: Args:
; REMARKS-NEXT:   - String:          EH Funclets are not supported yet.

; test @setjmp - similar to @catchret, but the MBB in question
; is the one generated when the setjmp's block is split

@buf = internal global [5 x ptr] zeroinitializer
declare ptr @llvm.frameaddress(i32) nounwind readnone
declare ptr @llvm.stacksave() nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) nounwind

define void @setjmp(i1 %b) nounwind {
entry:
  br i1 %b, label %early_out, label %sj
early_out:
  ret void
sj:
  %fp = call ptr @llvm.frameaddress(i32 0)
  store ptr %fp, ptr @buf, align 16
  %sp = call ptr @llvm.stacksave()
  store ptr %sp, ptr getelementptr inbounds ([5 x ptr], ptr @buf, i64 0, i64 2), align 16
  call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret void
}
; CHECK-LABEL: setjmp: # @setjmp
; CHECK: # %sj
; CHECK: leaq [[Label:\..+]](%rip), %[[Reg:.+]]{{$}}
; CHECK-NEXT: movq %[[Reg]], buf
; CHECK: {{^}}[[Label]]:  # Block address taken
; CHECK-NEXT:              # %sj
