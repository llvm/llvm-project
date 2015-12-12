; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s -enable-shrink-wrap=false | FileCheck %s
; Make sure shrink-wrapping does not break the lowering of exception handling.
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s -enable-shrink-wrap=true | FileCheck %s

; Repro cases from PR25168

; test @catchret - catchret target is not address-taken until PEI
; splits it into lea/mov followed by ret.  Make sure the MBB is
; handled, both by tempting BranchFolding to merge it with %early_out
; and delete it, and by checking that we emit a proper reference
; to it in the LEA

declare void @ProcessCLRException()
declare void @f()

define void @catchret(i1 %b) personality void ()* @ProcessCLRException {
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


; test @setjmp - similar to @catchret, but the MBB in question
; is the one generated when the setjmp's block is split

@buf = internal global [5 x i8*] zeroinitializer
declare i8* @llvm.frameaddress(i32) nounwind readnone
declare i8* @llvm.stacksave() nounwind
declare i32 @llvm.eh.sjlj.setjmp(i8*) nounwind
declare void @llvm.eh.sjlj.longjmp(i8*) nounwind

define void @setjmp(i1 %b) nounwind {
entry:
  br i1 %b, label %early_out, label %sj
early_out:
  ret void
sj:
  %fp = call i8* @llvm.frameaddress(i32 0)
  store i8* %fp, i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @buf, i64 0, i64 0), align 16
  %sp = call i8* @llvm.stacksave()
  store i8* %sp, i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @buf, i64 0, i64 2), align 16
  call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([5 x i8*]* @buf to i8*))
  ret void
}
; CHECK-LABEL: setjmp: # @setjmp
; CHECK: # %sj
; CHECK: leaq [[Label:\..+]](%rip), %[[Reg:.+]]{{$}}
; CHECK-NEXT: movq %[[Reg]], buf
; CHECK: {{^}}[[Label]]:  # Block address taken
; CHECK-NEXT:              # %sj
