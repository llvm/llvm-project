; RUN: opt %s -passes=sroa -o - -S \
; RUN: | FileCheck %s

;; Generated from this C++ source:
;; __attribute__((nodebug)) struct Blob {int P[6];} Glob;
;; __attribute__((nodebug)) int Cond;
;; __attribute__((nodebug)) Blob *C;
;; __attribute__((nodebug)) void call(int);
;; 
;; void f() {
;;   int A[16];
;;   __attribute__ ((nodebug)) int B[16];
;;   // A[0:6) <- Glob
;;   __builtin_memmove(&A[0], &Glob, sizeof(Blob));
;;   call(0);
;;   // B[8:14) <- Glob
;;   __builtin_memmove(&B[8], &Glob, sizeof(Blob));  
;;   call(A[0]);
;;   // A[8:14) <- A[0:6)
;;   __builtin_memmove(&A[8], &A[0], sizeof(Blob));
;;   call(A[8]);
;;   if (Cond)
;;     // C <- A[8:14)
;;     __builtin_memmove(C, &A[8], sizeof(Blob));
;;   else
;;     // C <- B[8:14)
;;     __builtin_memmove(C, &B[8], sizeof(Blob));    
;; }
;; 
;; using:
;;   clang test.cpp -emit-llvm -S -g -O2 -Xclang -disable-llvm-passes -o - \
;;   | opt -passes=declare-to-assign -o test.ll - -S

;; We're interested in variable A and the second memmove with A as a dest (the
;; third memmove in the source). SROA is going to chop up A so that the only
;; Alloca'd slice remaining is what were originally elements 1 through 5
;; inclusive (element 0 is promoted). Incidentally, the memmove later becomes a
;; memcpy. Check that the dbg.assign address and fragment are correct and
;; ensure the DIAssignID still links it to the memmove(/memcpy).

; CHECK: %A.sroa.0.sroa.5 = alloca [5 x i32]
; CHECK: llvm.memcpy{{.*}}(ptr align 4 %A.sroa.0.sroa.5, ptr align 4 getelementptr inbounds (i8, ptr @Glob, i64 4), i64 20, i1 false){{.*}}!DIAssignID ![[ID:[0-9]+]]
;; Here's the dbg.assign for element 0 - it's not important for the test.
; CHECK-NEXT: llvm.dbg.value({{.*}}!DIExpression(DW_OP_LLVM_fragment, 0, 32){{.*}})
;; This is the dbg.assign we care about:
; CHECK-NEXT: llvm.dbg.assign(metadata i1 undef, metadata ![[VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 160), metadata ![[ID]], metadata ptr %A.sroa.0.sroa.5, metadata !DIExpression())

; CHECK: ![[VAR]] = !DILocalVariable(name: "A"

source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Blob = type { [6 x i32] }

@Glob = dso_local global %struct.Blob zeroinitializer, align 4
@Cond = dso_local global i32 0, align 4
@C = dso_local global ptr null, align 8

; Function Attrs: mustprogress uwtable
define dso_local void @_Z1fv() #0 !dbg !9 {
entry:
  %A = alloca [16 x i32], align 16, !DIAssignID !18
  call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(), metadata !18, metadata ptr %A, metadata !DIExpression()), !dbg !19
  %B = alloca [16 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 64, ptr %A) #5, !dbg !20
  call void @llvm.lifetime.start.p0(i64 64, ptr %B) #5, !dbg !21
  %arrayidx = getelementptr inbounds [16 x i32], ptr %A, i64 0, i64 0, !dbg !22
  call void @llvm.memmove.p0.p0.i64(ptr align 16 %arrayidx, ptr align 4 @Glob, i64 24, i1 false), !dbg !23, !DIAssignID !24
  call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 192), metadata !24, metadata ptr %arrayidx, metadata !DIExpression()), !dbg !19
  call void @_Z4calli(i32 noundef 0), !dbg !25
  %arrayidx1 = getelementptr inbounds [16 x i32], ptr %B, i64 0, i64 8, !dbg !26
  call void @llvm.memmove.p0.p0.i64(ptr align 16 %arrayidx1, ptr align 4 @Glob, i64 24, i1 false), !dbg !27
  %arrayidx2 = getelementptr inbounds [16 x i32], ptr %A, i64 0, i64 0, !dbg !28
  %0 = load i32, ptr %arrayidx2, align 16, !dbg !28
  call void @_Z4calli(i32 noundef %0), !dbg !33
  %arrayidx3 = getelementptr inbounds [16 x i32], ptr %A, i64 0, i64 8, !dbg !34
  %arrayidx4 = getelementptr inbounds [16 x i32], ptr %A, i64 0, i64 0, !dbg !35
  call void @llvm.memmove.p0.p0.i64(ptr align 16 %arrayidx3, ptr align 16 %arrayidx4, i64 24, i1 false), !dbg !36, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 256, 192), metadata !37, metadata ptr %arrayidx3, metadata !DIExpression()), !dbg !19
  %arrayidx5 = getelementptr inbounds [16 x i32], ptr %A, i64 0, i64 8, !dbg !38
  %1 = load i32, ptr %arrayidx5, align 16, !dbg !38
  call void @_Z4calli(i32 noundef %1), !dbg !39
  %2 = load i32, ptr @Cond, align 4, !dbg !40
  %tobool = icmp ne i32 %2, 0, !dbg !40
  br i1 %tobool, label %if.then, label %if.else, !dbg !42

if.then:                                          ; preds = %entry
  %3 = load ptr, ptr @C, align 8, !dbg !43
  %arrayidx6 = getelementptr inbounds [16 x i32], ptr %A, i64 0, i64 8, !dbg !46
  call void @llvm.memmove.p0.p0.i64(ptr align 4 %3, ptr align 16 %arrayidx6, i64 24, i1 false), !dbg !47
  br label %if.end, !dbg !47

if.else:                                          ; preds = %entry
  %4 = load ptr, ptr @C, align 8, !dbg !48
  %arrayidx7 = getelementptr inbounds [16 x i32], ptr %B, i64 0, i64 8, !dbg !49
  call void @llvm.memmove.p0.p0.i64(ptr align 4 %4, ptr align 16 %arrayidx7, i64 24, i1 false), !dbg !50
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.lifetime.end.p0(i64 64, ptr %B) #5, !dbg !51
  call void @llvm.lifetime.end.p0(i64 64, ptr %A) #5, !dbg !51
  ret void, !dbg !51
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg) #3
declare void @_Z4calli(i32 noundef) #4
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !1000}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 16.0.0"}
!9 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !1, file: !1, line: 6, type: !10, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "A", scope: !9, file: !1, line: 7, type: !14)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 512, elements: !16)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DISubrange(count: 16)
!18 = distinct !DIAssignID()
!19 = !DILocation(line: 0, scope: !9)
!20 = !DILocation(line: 7, column: 3, scope: !9)
!21 = !DILocation(line: 8, column: 3, scope: !9)
!22 = !DILocation(line: 10, column: 22, scope: !9)
!23 = !DILocation(line: 10, column: 3, scope: !9)
!24 = distinct !DIAssignID()
!25 = !DILocation(line: 11, column: 3, scope: !9)
!26 = !DILocation(line: 13, column: 22, scope: !9)
!27 = !DILocation(line: 13, column: 3, scope: !9)
!28 = !DILocation(line: 14, column: 8, scope: !9)
!33 = !DILocation(line: 14, column: 3, scope: !9)
!34 = !DILocation(line: 16, column: 22, scope: !9)
!35 = !DILocation(line: 16, column: 29, scope: !9)
!36 = !DILocation(line: 16, column: 3, scope: !9)
!37 = distinct !DIAssignID()
!38 = !DILocation(line: 17, column: 8, scope: !9)
!39 = !DILocation(line: 17, column: 3, scope: !9)
!40 = !DILocation(line: 18, column: 7, scope: !41)
!41 = distinct !DILexicalBlock(scope: !9, file: !1, line: 18, column: 7)
!42 = !DILocation(line: 18, column: 7, scope: !9)
!43 = !DILocation(line: 20, column: 23, scope: !41)
!46 = !DILocation(line: 20, column: 27, scope: !41)
!47 = !DILocation(line: 20, column: 5, scope: !41)
!48 = !DILocation(line: 23, column: 23, scope: !41)
!49 = !DILocation(line: 23, column: 27, scope: !41)
!50 = !DILocation(line: 23, column: 5, scope: !41)
!51 = !DILocation(line: 24, column: 1, scope: !9)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
