; RUN: opt %s -passes=sroa -S -o - | FileCheck %s

;; Generated from this C++ source:
;; struct U { unsigned long w[2]; };
;; U assign(U a, U b, bool c) { a = c ? b : a; return a; }
;;
;; using:
;;   clang test.cpp -emit-llvm -S -g -O2 -Xclang -disable-llvm-passes -o - \
;;   | opt -passes=declare-to-assign -o test.ll - -S
;; and then replacing the conditional operator's branches and phi with a select,
;; as SimplifyCFG does before SROA.

;; FIXME: SROA leaves %a and %b on the stack because the select's user is a
;; memcpy. The memcpy keeps its DIAssignID and its dbg.assign for "a".

; CHECK: %a = alloca %struct.U
; CHECK: %[[SEL:.*]] = select i1 %loadedv, ptr %b, ptr %a
; CHECK: call void @llvm.memcpy{{.*}}(ptr align 8 %a, ptr align 8 %[[SEL]], i64 16, i1 false){{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i1 poison, ![[A:[0-9]+]], !DIExpression(), ![[ID]], ptr %a, !DIExpression(),
; CHECK: ![[A]] = !DILocalVariable(name: "a"

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.U = type { [2 x i64] }

; Function Attrs: mustprogress nounwind uwtable
define dso_local { i64, i64 } @_Z6assign1US_b(i64 %a.coerce0, i64 %a.coerce1, i64 %b.coerce0, i64 %b.coerce1, i1 noundef %c) #0 !dbg !15 {
entry:
  %retval = alloca %struct.U, align 8
  %a = alloca %struct.U, align 8, !DIAssignID !30
    #dbg_assign(i1 poison, !27, !DIExpression(), !30, ptr %a, !DIExpression(), !31)
  %b = alloca %struct.U, align 8, !DIAssignID !32
    #dbg_assign(i1 poison, !28, !DIExpression(), !32, ptr %b, !DIExpression(), !31)
  %c.addr = alloca i8, align 1, !DIAssignID !33
    #dbg_assign(i1 poison, !29, !DIExpression(), !33, ptr %c.addr, !DIExpression(), !31)
  %0 = getelementptr inbounds nuw { i64, i64 }, ptr %a, i32 0, i32 0
  store i64 %a.coerce0, ptr %0, align 8, !DIAssignID !34
    #dbg_assign(i64 %a.coerce0, !27, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !34, ptr %0, !DIExpression(), !31)
  %1 = getelementptr inbounds nuw { i64, i64 }, ptr %a, i32 0, i32 1
  store i64 %a.coerce1, ptr %1, align 8, !DIAssignID !35
    #dbg_assign(i64 %a.coerce1, !27, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !35, ptr %1, !DIExpression(), !31)
  %2 = getelementptr inbounds nuw { i64, i64 }, ptr %b, i32 0, i32 0
  store i64 %b.coerce0, ptr %2, align 8, !DIAssignID !36
    #dbg_assign(i64 %b.coerce0, !28, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !36, ptr %2, !DIExpression(), !31)
  %3 = getelementptr inbounds nuw { i64, i64 }, ptr %b, i32 0, i32 1
  store i64 %b.coerce1, ptr %3, align 8, !DIAssignID !37
    #dbg_assign(i64 %b.coerce1, !28, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !37, ptr %3, !DIExpression(), !31)
  %storedv = zext i1 %c to i8
  store i8 %storedv, ptr %c.addr, align 1, !tbaa !38, !DIAssignID !40
    #dbg_assign(i8 %storedv, !29, !DIExpression(), !40, ptr %c.addr, !DIExpression(), !31)
  %4 = load i8, ptr %c.addr, align 1, !dbg !41, !tbaa !38, !range !42, !noundef !43
  %loadedv = trunc i8 %4 to i1, !dbg !44
  %cond = select i1 %loadedv, ptr %b, ptr %a, !dbg !46
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a, ptr align 8 %cond, i64 16, i1 false), !dbg !47, !tbaa.struct !48, !DIAssignID !50
    #dbg_assign(i1 poison, !27, !DIExpression(), !50, ptr %a, !DIExpression(), !31)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %retval, ptr align 8 %a, i64 16, i1 false), !dbg !51, !tbaa.struct !48
  %coerce.dive = getelementptr inbounds nuw %struct.U, ptr %retval, i32 0, i32 0, !dbg !52
  %5 = load { i64, i64 }, ptr %coerce.dive, align 8, !dbg !53
  ret { i64, i64 } %5, !dbg !54
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!llvm.errno.tbaa = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "410177be0b8388a1fb946d3f9e042778")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"debugger-tuning", !"gdb"}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 24.0.0git"}
!10 = !{!11, !12, i64 0}
!11 = !{!"__libc_errno", !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}
!15 = distinct !DISubprogram(name: "assign", linkageName: "_Z6assign1US_b", scope: !1, file: !1, line: 2, type: !16, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !26, keyInstructions: true)
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !18, !18, !25}
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "U", file: !1, line: 1, size: 128, flags: DIFlagTypePassByValue, elements: !19, identifier: "_ZTS1U")
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "w", scope: !18, file: !1, line: 1, baseType: !21, size: 128)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 128, elements: !23)
!22 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!23 = !{!24}
!24 = !DISubrange(count: 2)
!25 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!26 = !{!27, !28, !29}
!27 = !DILocalVariable(name: "a", arg: 1, scope: !15, file: !1, line: 2, type: !18)
!28 = !DILocalVariable(name: "b", arg: 2, scope: !15, file: !1, line: 2, type: !18)
!29 = !DILocalVariable(name: "c", arg: 3, scope: !15, file: !1, line: 2, type: !25)
!30 = distinct !DIAssignID()
!31 = !DILocation(line: 0, scope: !15)
!32 = distinct !DIAssignID()
!33 = distinct !DIAssignID()
!34 = distinct !DIAssignID()
!35 = distinct !DIAssignID()
!36 = distinct !DIAssignID()
!37 = distinct !DIAssignID()
!38 = !{!39, !39, i64 0}
!39 = !{!"bool", !13, i64 0}
!40 = distinct !DIAssignID()
!41 = !DILocation(line: 2, column: 34, scope: !15, atomGroup: 1, atomRank: 3)
!42 = !{i8 0, i8 2}
!43 = !{}
!44 = !DILocation(line: 2, column: 34, scope: !15, atomGroup: 1, atomRank: 2)
!45 = !DILocation(line: 2, column: 34, scope: !15, atomGroup: 1, atomRank: 1)
!46 = !DILocation(line: 2, column: 34, scope: !15)
!47 = !DILocation(line: 2, column: 32, scope: !15, atomGroup: 2, atomRank: 1)
!48 = !{i64 0, i64 16, !49}
!49 = !{!13, !13, i64 0}
!50 = distinct !DIAssignID()
!51 = !DILocation(line: 2, column: 52, scope: !15, atomGroup: 3, atomRank: 1)
!52 = !DILocation(line: 2, column: 45, scope: !15)
!53 = !DILocation(line: 2, column: 45, scope: !15, atomGroup: 3, atomRank: 2)
!54 = !DILocation(line: 2, column: 45, scope: !15, atomGroup: 3, atomRank: 1)
