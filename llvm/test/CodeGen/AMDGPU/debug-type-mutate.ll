; RUN: llc -stop-after=codegenprepare < %s | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@0 = addrspace(4) constant [16 x i8] c"AAAAAAAAAAAAAAAA", align 16
@1 = addrspace(1) constant [16 x i8] c"AAAAAAAAAAAAAAAA", align 16

define void @func1(i32 %a0, i8 %a1, ptr %a2) #0 {
; CHECK: define void @func1(i32 %a0, i8 %a1, ptr %a2) #0 {
; CHECK-NEXT: %promoted = zext i32 %a0 to i64
; CHECK-NEXT: %vl0 = lshr i64 %promoted, 12
; CHECK-NEXT: #dbg_value(!DIArgList(i32 0, i64 %vl0), !4, !DIExpression(DIOpArg(1, i64), DIOpConvert(i32), DIOpConvert(i8), DIOpFragment(24, 8)), !9)
  %vl0 = lshr i32 %a0, 12
    #dbg_value(!DIArgList(i32 0, i32 %vl0), !4, !DIExpression(DIOpArg(1, i32), DIOpConvert(i8), DIOpFragment(24, 8)), !9)
  %op0 = zext nneg i32 %vl0 to i64
  %op1 = getelementptr inbounds nuw i8, ptr addrspace(4) @0, i64 %op0
  %op2 = load i8, ptr addrspace(4) %op1, align 1
  store i8 %op2, ptr %a2, align 1
  ret void
}

define void @func2(i32 %a0, i8 %a1, ptr %a2) #0 {
; CHECK: define void @func2(i32 %a0, i8 %a1, ptr %a2) #0 {
; CHECK-NEXT: %vl0 = lshr i32 %a0, 12
; CHECK-NEXT: #dbg_value(!DIArgList(i32 0, i32 %vl0), !4, !DIExpression(DIOpArg(1, i32), DIOpConvert(i8), DIOpFragment(24, 8)), !9)
  %vl0 = lshr i32 %a0, 12
    #dbg_value(!DIArgList(i32 0, i32 %vl0), !4, !DIExpression(DIOpArg(1, i32), DIOpConvert(i8), DIOpFragment(24, 8)), !9)
  %op0 = zext nneg i32 %vl0 to i64
  %op1 = getelementptr inbounds nuw i8, ptr addrspace(1) @1, i64 %op0
  %op2 = load i8, ptr addrspace(1) %op1, align 1
  store i8 %op2, ptr %a2, align 1
  ret void
}


attributes #0 = { "target-cpu"="gfx1201" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "-", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "aux32", scope: !5, file: !1, line: 1757, type: !8)
!5 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1754, type: !6, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!9 = !DILocation(line: 0, scope: !5)
