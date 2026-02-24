; RUN: llc < %s -mattr=+ptx72 | FileCheck %s
;
;; Test multiple inline calls at the same level - verifies that inlined_at information
;; is correctly emitted when multiple different functions (or multiple copies of the same function) are inlined into a single caller.
;
; __device__ __forceinline__ int foo(int a)
; {
;   if (a > 7)
;     return a*a;
;   return ++a;
; }
;
; __device__ __forceinline__ int baz(int a)
; {
;   if (a > 23)
;     return a*2;
;   return ++a;
; }
;
; __device__ int bar(int i, int j)
; {
;   return i + j;
; }
;
; __device__ int d;
;
; // inlining two different functions
; __global__ void kernel1(int x, int y)
; {
;   d = bar(foo(x), baz(y));
; }
;
; // inlining two copies of same function
; __global__ void kernel2(int x, int y)
; {
;   d = bar(foo(x), foo(y));
; }
;
; // inlining two different functions, extra computation in caller (y+x)
; __global__ void kernel3(int x, int y)
; {
;   d = bar(foo(x), baz(y + x));
; }
;
; // inlining two copies of same function, extra computation in caller (y+x)
; __global__ void kernel4(int x, int y)
; {
;   d = bar(foo(x), foo(y + x));
; }
;
; CHECK: .entry _Z7kernel1ii(
; CHECK: .loc [[FILENUM:[1-9]]] 25
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOONAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 25
; CHECK: .loc [[FILENUM:[1-9]]] 25
; CHECK: .loc [[FILENUM]] 10 {{[0-9]*}}, function_name [[BAZNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 25
; CHECK: .loc [[FILENUM:[1-9]]] 25
; CHECK: .loc [[FILENUM]] 17 {{[0-9]*}}, function_name [[BARNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 25
;
; CHECK: .entry _Z7kernel2ii(
; CHECK: .loc [[FILENUM:[1-9]]] 31
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 31
; CHECK: .loc [[FILENUM:[1-9]]] 31
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 31
; CHECK: .loc [[FILENUM:[1-9]]] 31
; CHECK: .loc [[FILENUM]] 17 {{[0-9]*}}, function_name [[BARNAME]], inlined_at [[FILENUM]] 31
;
; CHECK: .entry _Z7kernel3ii(
; CHECK: .loc [[FILENUM:[1-9]]] 37
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 37
; CHECK: .loc [[FILENUM:[1-9]]] 37
; CHECK: .loc [[FILENUM]] 10 {{[0-9]*}}, function_name [[BAZNAME]], inlined_at [[FILENUM]] 37
; CHECK: .loc [[FILENUM:[1-9]]] 37
; CHECK: .loc [[FILENUM]] 17 {{[0-9]*}}, function_name [[BARNAME]], inlined_at [[FILENUM]] 37
;
; CHECK: .entry _Z7kernel4ii(
; CHECK: .loc [[FILENUM:[1-9]]] 43
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 43
; CHECK: .loc [[FILENUM:[1-9]]] 43
; CHECK: .loc [[FILENUM]] 3 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 43
; CHECK: .loc [[FILENUM:[1-9]]] 43
; CHECK: .loc [[FILENUM]] 17 {{[0-9]*}}, function_name [[BARNAME]], inlined_at [[FILENUM]] 43
; CHECK: .section .debug_str
; CHECK: {
; CHECK: [[FOONAME]]:
; CHECK-NEXT: // {{.*}} _Z3fooi
; CHECK: [[BAZNAME]]:
; CHECK-NEXT: // {{.*}} _Z3bazi
; CHECK: [[BARNAME]]:
; CHECK-NEXT: // {{.*}} _Z3barii
; CHECK: }

source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

@d = internal addrspace(1) global i32 0, align 4
@llvm.used = appending global [5 x ptr] [ptr @_Z7kernel1ii, ptr @_Z7kernel2ii, ptr @_Z7kernel3ii, ptr @_Z7kernel4ii, ptr addrspacecast (ptr addrspace(1) @d to ptr)], section "llvm.metadata"

define void @_Z7kernel1ii(i32 noundef %x, i32 noundef %y) !dbg !11 {
entry:
  %cmp.i = icmp sgt i32 %x, 7, !dbg !13
  %mul.i = mul nsw i32 %x, %x, !dbg !13
  %inc.i = add nsw i32 %x, 1, !dbg !13
  %retval.0.i = select i1 %cmp.i, i32 %mul.i, i32 %inc.i, !dbg !13
  %cmp.i1 = icmp sgt i32 %y, 23, !dbg !18
  %mul.i2 = shl nuw nsw i32 %y, 1, !dbg !18
  %inc.i3 = add nsw i32 %y, 1, !dbg !18
  %retval.0.i4 = select i1 %cmp.i1, i32 %mul.i2, i32 %inc.i3, !dbg !18
  %add.i = add nsw i32 %retval.0.i4, %retval.0.i, !dbg !22
  store i32 %add.i, ptr addrspace(1) @d, align 4, !dbg !26
  ret void, !dbg !27
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none)
define void @_Z7kernel2ii(i32 noundef %x, i32 noundef %y) #0 !dbg !28 {
entry:
  %cmp.i = icmp sgt i32 %x, 7, !dbg !29
  %mul.i = mul nsw i32 %x, %x, !dbg !29
  %inc.i = add nsw i32 %x, 1, !dbg !29
  %retval.0.i = select i1 %cmp.i, i32 %mul.i, i32 %inc.i, !dbg !29
  %cmp.i1 = icmp sgt i32 %y, 7, !dbg !32
  %mul.i2 = mul nsw i32 %y, %y, !dbg !32
  %inc.i3 = add nsw i32 %y, 1, !dbg !32
  %retval.0.i4 = select i1 %cmp.i1, i32 %mul.i2, i32 %inc.i3, !dbg !32
  %add.i = add nsw i32 %retval.0.i4, %retval.0.i, !dbg !34
  store i32 %add.i, ptr addrspace(1) @d, align 4, !dbg !36
  ret void, !dbg !37
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none)
define void @_Z7kernel3ii(i32 noundef %x, i32 noundef %y) #0 !dbg !38 {
entry:
  %cmp.i = icmp sgt i32 %x, 7, !dbg !39
  %mul.i = mul nsw i32 %x, %x, !dbg !39
  %inc.i = add nsw i32 %x, 1, !dbg !39
  %retval.0.i = select i1 %cmp.i, i32 %mul.i, i32 %inc.i, !dbg !39
  %add = add nsw i32 %y, %x, !dbg !42
  %cmp.i1 = icmp sgt i32 %add, 23, !dbg !43
  %mul.i2 = shl nuw nsw i32 %add, 1, !dbg !43
  %inc.i3 = add nsw i32 %add, 1, !dbg !43
  %retval.0.i4 = select i1 %cmp.i1, i32 %mul.i2, i32 %inc.i3, !dbg !43
  %add.i = add nsw i32 %retval.0.i4, %retval.0.i, !dbg !45
  store i32 %add.i, ptr addrspace(1) @d, align 4, !dbg !42
  ret void, !dbg !47
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none)
define void @_Z7kernel4ii(i32 noundef %x, i32 noundef %y) #0 !dbg !48 {
entry:
  %cmp.i = icmp sgt i32 %x, 7, !dbg !49
  %mul.i = mul nsw i32 %x, %x, !dbg !49
  %inc.i = add nsw i32 %x, 1, !dbg !49
  %retval.0.i = select i1 %cmp.i, i32 %mul.i, i32 %inc.i, !dbg !49
  %add = add nsw i32 %y, %x, !dbg !52
  %cmp.i1 = icmp sgt i32 %add, 7, !dbg !53
  %mul.i2 = mul nsw i32 %add, %add, !dbg !53
  %inc.i3 = add nsw i32 %add, 1, !dbg !53
  %retval.0.i4 = select i1 %cmp.i1, i32 %mul.i2, i32 %inc.i3, !dbg !53
  %add.i = add nsw i32 %retval.0.i4, %retval.0.i, !dbg !55
  store i32 %add.i, ptr addrspace(1) @d, align 4, !dbg !52
  ret void, !dbg !57
}

!llvm.dbg.cu = !{!0}
!nvvm.annotations = !{!5, !6, !7, !8}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "t4.cu", directory: "")
!5 = !{ptr @_Z7kernel1ii, !"kernel", i32 1}
!6 = !{ptr @_Z7kernel2ii, !"kernel", i32 1}
!7 = !{ptr @_Z7kernel3ii, !"kernel", i32 1}
!8 = !{ptr @_Z7kernel4ii, !"kernel", i32 1}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{}
!11 = distinct !DISubprogram(name: "kernel1", linkageName: "_Z7kernel1ii", scope: !1, file: !1, line: 23, type: !12, scopeLine: 23, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !10)
!13 = !DILocation(line: 3, column: 3, scope: !14, inlinedAt: !16)
!14 = distinct !DILexicalBlock(scope: !15, file: !1, line: 2, column: 1)
!15 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = distinct !DILocation(line: 25, column: 3, scope: !17)
!17 = distinct !DILexicalBlock(scope: !11, file: !1, line: 24, column: 1)
!18 = !DILocation(line: 10, column: 3, scope: !19, inlinedAt: !21)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 9, column: 1)
!20 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazi", scope: !1, file: !1, line: 8, type: !12, scopeLine: 8, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = distinct !DILocation(line: 25, column: 3, scope: !17)
!22 = !DILocation(line: 17, column: 3, scope: !23, inlinedAt: !25)
!23 = distinct !DILexicalBlock(scope: !24, file: !1, line: 16, column: 1)
!24 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barii", scope: !1, file: !1, line: 15, type: !12, scopeLine: 15, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!25 = distinct !DILocation(line: 25, column: 3, scope: !17)
!26 = !DILocation(line: 25, column: 3, scope: !17)
!27 = !DILocation(line: 26, column: 1, scope: !17)
!28 = distinct !DISubprogram(name: "kernel2", linkageName: "_Z7kernel2ii", scope: !1, file: !1, line: 29, type: !12, scopeLine: 29, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!29 = !DILocation(line: 3, column: 3, scope: !14, inlinedAt: !30)
!30 = distinct !DILocation(line: 31, column: 3, scope: !31)
!31 = distinct !DILexicalBlock(scope: !28, file: !1, line: 30, column: 1)
!32 = !DILocation(line: 3, column: 3, scope: !14, inlinedAt: !33)
!33 = distinct !DILocation(line: 31, column: 3, scope: !31)
!34 = !DILocation(line: 17, column: 3, scope: !23, inlinedAt: !35)
!35 = distinct !DILocation(line: 31, column: 3, scope: !31)
!36 = !DILocation(line: 31, column: 3, scope: !31)
!37 = !DILocation(line: 32, column: 1, scope: !31)
!38 = distinct !DISubprogram(name: "kernel3", linkageName: "_Z7kernel3ii", scope: !1, file: !1, line: 35, type: !12, scopeLine: 35, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!39 = !DILocation(line: 3, column: 3, scope: !14, inlinedAt: !40)
!40 = distinct !DILocation(line: 37, column: 3, scope: !41)
!41 = distinct !DILexicalBlock(scope: !38, file: !1, line: 36, column: 1)
!42 = !DILocation(line: 37, column: 3, scope: !41)
!43 = !DILocation(line: 10, column: 3, scope: !19, inlinedAt: !44)
!44 = distinct !DILocation(line: 37, column: 3, scope: !41)
!45 = !DILocation(line: 17, column: 3, scope: !23, inlinedAt: !46)
!46 = distinct !DILocation(line: 37, column: 3, scope: !41)
!47 = !DILocation(line: 38, column: 1, scope: !41)
!48 = distinct !DISubprogram(name: "kernel4", linkageName: "_Z7kernel4ii", scope: !1, file: !1, line: 41, type: !12, scopeLine: 41, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!49 = !DILocation(line: 3, column: 3, scope: !14, inlinedAt: !50)
!50 = distinct !DILocation(line: 43, column: 3, scope: !51)
!51 = distinct !DILexicalBlock(scope: !48, file: !1, line: 42, column: 1)
!52 = !DILocation(line: 43, column: 3, scope: !51)
!53 = !DILocation(line: 3, column: 3, scope: !14, inlinedAt: !54)
!54 = distinct !DILocation(line: 43, column: 3, scope: !51)
!55 = !DILocation(line: 17, column: 3, scope: !23, inlinedAt: !56)
!56 = distinct !DILocation(line: 43, column: 3, scope: !51)
!57 = !DILocation(line: 44, column: 1, scope: !51)
