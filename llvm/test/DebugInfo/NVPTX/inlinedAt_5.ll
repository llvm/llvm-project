; RUN: llc < %s -mattr=+ptx72 | FileCheck %s
;
;; Test deep inline chain - verifies that inlined_at information is correctly emitted
;; through an 11-level deep chain of inlining (foo0 through foo10).
;
; __device__ int foo0(int a, int b) {
;   if (a > b)
;     return --a;
;   if (a > b - 7)
;     return a*2;
;   return ++a;
; }
; __device__ int foo1(int a, int b) {
;   return foo0(a*3, b*b);
; }
; __device__ int foo2(int a, int b) {
;   return foo1(a+2, b*b);
; }
; __device__ int foo3(int a, int b) {
;   return foo2(a+100, b-7);
; }
; __device__ int foo4(int a, int b) {
;   return foo3(a*a, b*3);
; }
; __device__ int foo5(int a, int b) {
;   return foo4(a*3, b*3);
; }
; __device__ int foo6(int a, int b) {
;   return foo5(a*3, b*3);
; }
; __device__ int foo7(int a, int b) {
;   return foo6(a*a + 2, b*b*5);
; }
; __device__ int foo8(int a, int b) {
;   return foo7(a*2, b*2*a);
; }
; __device__ int foo9(int a, int b) {
;   return foo8(a*2, b*2*a);
; }
; __device__ int foo10(int a, int b) {
;   return foo9(a*2*b, b*2);
; }
;
; __device__ int g;
;
; __global__ void kernel(int a, int b) {
;   g = foo10(a, b);
; }
;
; CHECK: .entry _Z6kernelii(
; CHECK: .loc [[FILENUM:[1-9]]] 42
; CHECK: .loc [[FILENUM]] 36 {{[0-9]*}}, function_name [[FOO10NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 42
; CHECK: .loc [[FILENUM]] 33 {{[0-9]*}}, function_name [[FOO9NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 36
; CHECK: .loc [[FILENUM]] 30 {{[0-9]*}}, function_name [[FOO8NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 33
; CHECK: .loc [[FILENUM]] 27 {{[0-9]*}}, function_name [[FOO7NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 30
; CHECK: .loc [[FILENUM]] 24 {{[0-9]*}}, function_name [[FOO6NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 27
; CHECK: .loc [[FILENUM]] 21 {{[0-9]*}}, function_name [[FOO5NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 24
; CHECK: .loc [[FILENUM]] 18 {{[0-9]*}}, function_name [[FOO4NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 21
; CHECK: .loc [[FILENUM]] 15 {{[0-9]*}}, function_name [[FOO3NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 18
; CHECK: .loc [[FILENUM]] 12 {{[0-9]*}}, function_name [[FOO2NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 15
; CHECK: .loc [[FILENUM]] 9 {{[0-9]*}}, function_name [[FOO1NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 12
; CHECK: .loc [[FILENUM]] 2 {{[0-9]*}}, function_name [[FOO0NAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 9
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOO0NAME]], inlined_at [[FILENUM]] 9
; CHECK: .section .debug_str
; CHECK: {
; CHECK: [[FOO10NAME]]:
; CHECK-NEXT: // {{.*}} _Z5foo10ii
; CHECK: [[FOO9NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo9ii
; CHECK: [[FOO8NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo8ii
; CHECK: [[FOO7NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo7ii
; CHECK: [[FOO6NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo6ii
; CHECK: [[FOO5NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo5ii
; CHECK: [[FOO4NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo4ii
; CHECK: [[FOO3NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo3ii
; CHECK: [[FOO2NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo2ii
; CHECK: [[FOO1NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo1ii
; CHECK: [[FOO0NAME]]:
; CHECK-NEXT: // {{.*}} _Z4foo0ii
; CHECK: }

source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

@g = internal addrspace(1) global i32 0, align 4
@llvm.used = appending global [2 x ptr] [ptr @_Z6kernelii, ptr addrspacecast (ptr addrspace(1) @g to ptr)], section "llvm.metadata"

define void @_Z6kernelii(i32 noundef %a, i32 noundef %b) !dbg !5 {
entry:
  %mul.i = shl nsw i32 %a, 1, !dbg !7
  %mul2.i = mul nsw i32 %mul.i, %b, !dbg !7
  %mul.i.i = shl nsw i32 %mul2.i, 1, !dbg !12
  %mul.i.i.i = shl nsw i32 %mul2.i, 2, !dbg !16
  %mul4.i.i = shl i32 %b, 3, !dbg !12
  %mul2.i.i.i = mul i32 %mul4.i.i, %mul2.i, !dbg !16
  %mul4.i.i.i = mul nsw i32 %mul2.i.i.i, %mul.i.i, !dbg !16
  %mul.i.i.i.i = mul i32 %mul2.i, 36, !dbg !20
  %0 = mul i32 %mul.i.i.i.i, %mul.i.i.i, !dbg !24
  %mul.i.i.i.i.i.i = add nuw nsw i32 %0, 18, !dbg !24
  %mul4.i.i.i.i = mul i32 %mul4.i.i.i, 135, !dbg !20
  %mul3.i.i.i.i.i.i.i = mul i32 %mul4.i.i.i.i, %mul4.i.i.i, !dbg !31
  %sub.i.i.i.i.i.i.i.i = add nsw i32 %mul3.i.i.i.i.i.i.i, -7, !dbg !35
  %mul.i.i.i.i.i.i.i.i.i = mul nsw i32 %sub.i.i.i.i.i.i.i.i, %sub.i.i.i.i.i.i.i.i, !dbg !39
  %mul.i.i.i.i.i.i.i = mul i32 %mul.i.i.i.i.i.i, 3, !dbg !31
  %1 = mul i32 %mul.i.i.i.i.i.i.i, %mul.i.i.i.i.i.i, !dbg !43
  %mul.i.i.i.i.i.i.i.i.i.i = add i32 %1, 306, !dbg !43
  %mul3.i.i.i.i.i.i.i.i.i.i = mul nuw nsw i32 %mul.i.i.i.i.i.i.i.i.i, %mul.i.i.i.i.i.i.i.i.i, !dbg !43
  %cmp.i.i.i.i.i.i.i.i.i.i.i = icmp sgt i32 %mul.i.i.i.i.i.i.i.i.i.i, %mul3.i.i.i.i.i.i.i.i.i.i, !dbg !47
  br i1 %cmp.i.i.i.i.i.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i.i.i.i, label %if.end.i.i.i.i.i.i.i.i.i.i.i, !dbg !47

if.then.i.i.i.i.i.i.i.i.i.i.i:                    ; preds = %entry
  %dec.i.i.i.i.i.i.i.i.i.i.i = add i32 %1, 305, !dbg !51
  br label %_Z5foo10ii.exit, !dbg !51

if.end.i.i.i.i.i.i.i.i.i.i.i:                     ; preds = %entry
  %sub.i.i.i.i.i.i.i.i.i.i.i = add nsw i32 %mul3.i.i.i.i.i.i.i.i.i.i, -7, !dbg !53
  %cmp5.i.i.i.i.i.i.i.i.i.i.i = icmp sgt i32 %mul.i.i.i.i.i.i.i.i.i.i, %sub.i.i.i.i.i.i.i.i.i.i.i, !dbg !53
  br i1 %cmp5.i.i.i.i.i.i.i.i.i.i.i, label %if.then6.i.i.i.i.i.i.i.i.i.i.i, label %if.end8.i.i.i.i.i.i.i.i.i.i.i, !dbg !53

if.then6.i.i.i.i.i.i.i.i.i.i.i:                   ; preds = %if.end.i.i.i.i.i.i.i.i.i.i.i
  %mul.i.i.i.i.i.i.i.i.i.i.i = shl nsw i32 %mul.i.i.i.i.i.i.i.i.i.i, 1, !dbg !54
  br label %_Z5foo10ii.exit, !dbg !54

if.end8.i.i.i.i.i.i.i.i.i.i.i:                    ; preds = %if.end.i.i.i.i.i.i.i.i.i.i.i
  %inc.i.i.i.i.i.i.i.i.i.i.i = add i32 %1, 307, !dbg !56
  br label %_Z5foo10ii.exit, !dbg !56

_Z5foo10ii.exit:                                  ; preds = %if.then.i.i.i.i.i.i.i.i.i.i.i, %if.then6.i.i.i.i.i.i.i.i.i.i.i, %if.end8.i.i.i.i.i.i.i.i.i.i.i
  %retval.0.i.i.i.i.i.i.i.i.i.i.i = phi i32 [ %dec.i.i.i.i.i.i.i.i.i.i.i, %if.then.i.i.i.i.i.i.i.i.i.i.i ], [ %mul.i.i.i.i.i.i.i.i.i.i.i, %if.then6.i.i.i.i.i.i.i.i.i.i.i ], [ %inc.i.i.i.i.i.i.i.i.i.i.i, %if.end8.i.i.i.i.i.i.i.i.i.i.i ], !dbg !56
  store i32 %retval.0.i.i.i.i.i.i.i.i.i.i.i, ptr addrspace(1) @g, align 4, !dbg !57
  ret void, !dbg !58
}

!llvm.dbg.cu = !{!0}
!nvvm.annotations = !{!3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "t5.cu", directory: "")
!2 = !{}
!3 = !{ptr @_Z6kernelii, !"kernel", i32 1}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kernelii", scope: !1, file: !1, line: 41, type: !6, scopeLine: 41, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 36, column: 3, scope: !8, inlinedAt: !10)
!8 = distinct !DILexicalBlock(scope: !9, file: !1, line: 35, column: 28)
!9 = distinct !DISubprogram(name: "foo10", linkageName: "_Z5foo10ii", scope: !1, file: !1, line: 35, type: !6, scopeLine: 35, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = distinct !DILocation(line: 42, column: 3, scope: !11)
!11 = distinct !DILexicalBlock(scope: !5, file: !1, line: 41, column: 29)
!12 = !DILocation(line: 33, column: 3, scope: !13, inlinedAt: !15)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 32, column: 28)
!14 = distinct !DISubprogram(name: "foo9", linkageName: "_Z4foo9ii", scope: !1, file: !1, line: 32, type: !6, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = distinct !DILocation(line: 36, column: 3, scope: !8, inlinedAt: !10)
!16 = !DILocation(line: 30, column: 3, scope: !17, inlinedAt: !19)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 29, column: 28)
!18 = distinct !DISubprogram(name: "foo8", linkageName: "_Z4foo8ii", scope: !1, file: !1, line: 29, type: !6, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!19 = distinct !DILocation(line: 33, column: 3, scope: !13, inlinedAt: !15)
!20 = !DILocation(line: 27, column: 3, scope: !21, inlinedAt: !23)
!21 = distinct !DILexicalBlock(scope: !22, file: !1, line: 26, column: 28)
!22 = distinct !DISubprogram(name: "foo7", linkageName: "_Z4foo7ii", scope: !1, file: !1, line: 26, type: !6, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = distinct !DILocation(line: 30, column: 3, scope: !17, inlinedAt: !19)
!24 = !DILocation(line: 21, column: 3, scope: !25, inlinedAt: !27)
!25 = distinct !DILexicalBlock(scope: !26, file: !1, line: 20, column: 28)
!26 = distinct !DISubprogram(name: "foo5", linkageName: "_Z4foo5ii", scope: !1, file: !1, line: 20, type: !6, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!27 = distinct !DILocation(line: 24, column: 3, scope: !28, inlinedAt: !30)
!28 = distinct !DILexicalBlock(scope: !29, file: !1, line: 23, column: 28)
!29 = distinct !DISubprogram(name: "foo6", linkageName: "_Z4foo6ii", scope: !1, file: !1, line: 23, type: !6, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!30 = distinct !DILocation(line: 27, column: 3, scope: !21, inlinedAt: !23)
!31 = !DILocation(line: 18, column: 3, scope: !32, inlinedAt: !34)
!32 = distinct !DILexicalBlock(scope: !33, file: !1, line: 17, column: 28)
!33 = distinct !DISubprogram(name: "foo4", linkageName: "_Z4foo4ii", scope: !1, file: !1, line: 17, type: !6, scopeLine: 17, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!34 = distinct !DILocation(line: 21, column: 3, scope: !25, inlinedAt: !27)
!35 = !DILocation(line: 15, column: 3, scope: !36, inlinedAt: !38)
!36 = distinct !DILexicalBlock(scope: !37, file: !1, line: 14, column: 28)
!37 = distinct !DISubprogram(name: "foo3", linkageName: "_Z4foo3ii", scope: !1, file: !1, line: 14, type: !6, scopeLine: 14, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!38 = distinct !DILocation(line: 18, column: 3, scope: !32, inlinedAt: !34)
!39 = !DILocation(line: 12, column: 3, scope: !40, inlinedAt: !42)
!40 = distinct !DILexicalBlock(scope: !41, file: !1, line: 11, column: 28)
!41 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2ii", scope: !1, file: !1, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!42 = distinct !DILocation(line: 15, column: 3, scope: !36, inlinedAt: !38)
!43 = !DILocation(line: 9, column: 3, scope: !44, inlinedAt: !46)
!44 = distinct !DILexicalBlock(scope: !45, file: !1, line: 8, column: 28)
!45 = distinct !DISubprogram(name: "foo1", linkageName: "_Z4foo1ii", scope: !1, file: !1, line: 8, type: !6, scopeLine: 8, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!46 = distinct !DILocation(line: 12, column: 3, scope: !40, inlinedAt: !42)
!47 = !DILocation(line: 2, column: 3, scope: !48, inlinedAt: !50)
!48 = distinct !DILexicalBlock(scope: !49, file: !1, line: 1, column: 28)
!49 = distinct !DISubprogram(name: "foo0", linkageName: "_Z4foo0ii", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!50 = distinct !DILocation(line: 9, column: 3, scope: !44, inlinedAt: !46)
!51 = !DILocation(line: 3, column: 5, scope: !52, inlinedAt: !50)
!52 = distinct !DILexicalBlock(scope: !48, file: !1, line: 2, column: 3)
!53 = !DILocation(line: 4, column: 3, scope: !48, inlinedAt: !50)
!54 = !DILocation(line: 5, column: 5, scope: !55, inlinedAt: !50)
!55 = distinct !DILexicalBlock(scope: !48, file: !1, line: 4, column: 3)
!56 = !DILocation(line: 6, column: 3, scope: !48, inlinedAt: !50)
!57 = !DILocation(line: 42, column: 3, scope: !11)
!58 = !DILocation(line: 43, column: 1, scope: !11)
