; RUN: llc -filetype=asm -code-model=medium %s --large-data-threshold=65636 -o - | FileCheck %s --check-prefix=CHECKASM-MEDIUM
; RUN: llc -filetype=asm -code-model=large %s -o - | FileCheck %s --check-prefix=CHECKASM-LARGE

; CHECKASM-MEDIUM:       .section	.lbss,"awl",@nobits
; CHECKASM-MEDIUM-NEXT:	 .type	__BLNK__,@object                # @__BLNK__
; CHECKASM-MEDIUM-NEXT:  .largecomm	__BLNK__,48394093832,8
; CHECKASM-MEDIUM-NEXT:  .type	ccc_,@object                    # @ccc_
; CHECKASM-MEDIUM-NEXT:  .comm	ccc_,8,8

; CHECKASM-LARGE:       .section	.lbss,"awl",@nobits
; CHECKASM-LARGE-NEXT:	.type	__BLNK__,@object                # @__BLNK__
; CHECKASM-LARGE-NEXT:  .largecomm	__BLNK__,48394093832,8
; CHECKASM-LARGE-NEXT:  .type	ccc_,@object                    # @ccc_
; CHECKASM-LARGE-NEXT:  .largecomm	ccc_,8,8

source_filename = "FIRModule"
target triple = "x86_64-unknown-linux-gnu"

@__BLNK__ = common global [48394093832 x i8] zeroinitializer, align 8
@ccc_ = common global [8 x i8] zeroinitializer, align 8
@_QFECn1 = internal constant i32 77777
@_QFECn2 = internal constant i32 77777

define void @_QQmain() #0 {
  store double 1.000000e+00, ptr @ccc_, align 8
  store double 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @__BLNK__, i64 61600176), align 8
  ret void
}

declare void @_FortranAProgramStart(i32, ptr, ptr, ptr) #1

declare void @_FortranAProgramEndStatement() #1

define i32 @main(i32 %0, ptr %1, ptr %2) #0 {
  call void @_FortranAProgramStart(i32 %0, ptr %1, ptr %2, ptr null)
  call void @_QQmain()
  call void @_FortranAProgramEndStatement()
  ret i32 0
}

attributes #0 = { "frame-pointer"="all" "target-cpu"="x86-64" }
attributes #1 = { "frame-pointer"="all" }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3}

!0 = !{!"flang version 22.0.0 (https://github.com/llvm/llvm-project.git e1afe25356b8d2ee14f5f88bdb6c2a1526ed14ef)"}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
