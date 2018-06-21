; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; void usevars(int, ...);
; void f() {
;   int a[5] = {9, 4, 5, 4, 2};
;   usevars(a[0], a);
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1001)
; CHECK:     Name: f
; CHECK:   }
; CHECK:   Array (0x1003) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: int (0x74)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 20
; CHECK:     Name:
; CHECK:   }
; CHECK: ]
; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S (5)
; CHECK:   Magic: 0x4
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     {{.*}}Proc{{.*}}Sym {
; CHECK:       PtrParent: 0x0
; CHECK:       PtrEnd: 0x0
; CHECK:       PtrNext: 0x0
; CHECK:       CodeSize: 0x39
; CHECK:       DbgStart: 0x0
; CHECK:       DbgEnd: 0x0
; CHECK:       FunctionType: f (0x1002)
; CHECK:       CodeOffset: ?f@@YAXXZ+0x0
; CHECK:       Segment: 0x0
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       DisplayName: f
; CHECK:       LinkageName: ?f@@YAXXZ
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: 0x1003
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: a
; CHECK:     }
; CHECK:     DefRangeRegisterRelSym {
; CHECK:       BaseRegister: EBP (0x16)
; CHECK:       HasSpilledUDTMember: No
; CHECK:       OffsetInParent: 0
; CHECK:       BasePointerOffset: -20
; CHECK:       LocalVariableAddrRange {
; CHECK:         OffsetStart: .text+0x3
; CHECK:         ISectStart: 0x0
; CHECK:         Range: 0x36
; CHECK:       }
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:     }
; CHECK:   ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.23918"

@"\01?a@?1??f@@YAXXZ@3PAHA" = private unnamed_addr constant [5 x i32] [i32 9, i32 4, i32 5, i32 4, i32 2], align 4

define void @"\01?f@@YAXXZ"() #0 !dbg !6 {
entry:
  %a = alloca [5 x i32], align 4
  call void @llvm.dbg.declare(metadata [5 x i32]* %a, metadata !9, metadata !14), !dbg !15
  %0 = bitcast [5 x i32]* %a to i8*, !dbg !15
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* bitcast ([5 x i32]* @"\01?a@?1??f@@YAXXZ@3PAHA" to i8*), i32 20, i32 4, i1 false), !dbg !15
  %arraydecay = getelementptr inbounds [5 x i32], [5 x i32]* %a, i32 0, i32 0, !dbg !16
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %a, i32 0, i32 0, !dbg !17
  %1 = load i32, i32* %arrayidx, align 4, !dbg !17
  call void (i32, ...) @"\01?usevars@@YAXHZZ"(i32 %1, i32* %arraydecay), !dbg !18
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) #2

declare void @"\01?usevars@@YAXHZZ"(i32, ...) #3

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cllvm\5Ctest\5CDebugInfo\5CCOFF")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 3, type: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 160, align: 32, elements: !12)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(count: 5)
!14 = !DIExpression()
!15 = !DILocation(line: 3, column: 7, scope: !6)
!16 = !DILocation(line: 4, column: 17, scope: !6)
!17 = !DILocation(line: 4, column: 11, scope: !6)
!18 = !DILocation(line: 4, column: 3, scope: !6)
!19 = !DILocation(line: 5, column: 1, scope: !6)
