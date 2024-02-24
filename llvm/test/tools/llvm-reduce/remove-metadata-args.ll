; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=metadata --test FileCheck --test-arg %s --test-arg --check-prefix=EXCITING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; All exciting stuff must remain in the reduced file.
; EXCITING-DAG: ExcitingGlobal = global i32 0, !md !0
; EXCITING-DAG: define void @ExcitingFunc() !md !0
; EXCITING-DAG: store i32 0, ptr @ExcitingGlobal, align 4, !md !0
; EXCITING-DAG: !ExcitingNamedMD = !{!0}

; Boring stuff's metadata must have been removed.
; REDUCED-NOT: Boring{{.*}} !md !0
; REDUCED-NOT: !md !0 {{.*}}Boring


@ExcitingGlobal = global i32 0, !md !0
@BoringGlobal = global i32 0, !md !0

define void @ExcitingFunc() !md !0 {
   store i32 0, ptr @ExcitingGlobal, align 4, !md !0
   store i32 0, ptr @BoringGlobal, align 4, !md !0
   ret void
}

declare !md !0 void @BoringFunc()

!ExcitingNamedMD = !{!0}
!BoringNamedMD = !{!0}

!0 = !{!"my metadata"}
