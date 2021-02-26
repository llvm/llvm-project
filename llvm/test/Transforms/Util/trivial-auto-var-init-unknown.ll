; RUN: opt -annotation-remarks -o /dev/null -S -pass-remarks-output=%t.opt.yaml %s -pass-remarks-missed=annotation-remarks 2>&1 | FileCheck %s
; RUN: cat %t.opt.yaml | FileCheck -check-prefix=YAML %s

; No remarks for this function, no instructions with metadata.
define void @none() {
; YAML-NOT:  Function:        none
  ret void
}

; Emit a remark that reports an instruction we can't analyze.
define void @unknown() {
; CHECK: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        unknown
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Initialization inserted by -ftrivial-auto-var-init.
; YAML-NEXT: ...
  ret void, !annotation !0, !dbg !DILocation(scope: !4)
}

; Emit a remark that reports an intrinsic call to an unknown intrinsic.
define void @unknown_intrinsic(i32* %dst) {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        unknown_intrinsic
  call i8* @llvm.returnaddress(i32 0), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a function call to a known libcall.
define void @known_call(i8* %dst) {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        known_call
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Initialization inserted by -ftrivial-auto-var-init.
; YAML-NEXT: ...
  call i8* @memset(i8* %dst, i32 0, i64 32), !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
declare i8* @memset(i8*, i32, i64)

!llvm.module.flags = !{!1}
!0 = !{ !"auto-init" }
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
!3 = !DIFile(filename: "file", directory: "")
!4 = distinct !DISubprogram(name: "function", scope: !3, file: !3, unit: !2)
!5 = !DIBasicType(name: "byte", size: 8)
!6 = !DILocalVariable(name: "destination", scope: !4, file: !3, type: !5)
!7 = !DILocalVariable(name: "destination2", scope: !4, file: !3, type: !5)
