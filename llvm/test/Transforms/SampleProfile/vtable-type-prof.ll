; Tests that the sample profile loader annotates vtable type profiles (recorded
; in the sample profile as `vtables` lines) as IPVK_VTableTarget value profile
; metadata on the vtable load instruction feeding a virtual call, and that the
; indirect call promotion pass can consume it for vtable-based comparison.

; Without -enable-vtable-profile-use, vtable loads are left alone.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/vtable-type-prof.prof -S | FileCheck %s --check-prefixes=COMMON,NOVTABLE
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/vtable-type-prof.prof -enable-vtable-profile-use -S | FileCheck %s --check-prefixes=COMMON,VTABLE
; Only the top -icp-max-num-vtables entries are annotated.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/vtable-type-prof.prof -enable-vtable-profile-use -icp-max-num-vtables=1 -S | FileCheck %s --check-prefixes=COMMON,VTABLE1
; Stale vtable metadata is dropped for cold virtual calls.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/vtable-type-prof.prof -enable-vtable-profile-use -overwrite-existing-weights -S | FileCheck %s --check-prefixes=COMMON,VTABLE,OVERWRITE
; End to end: ICP compares the vtable instead of the function pointer.
; RUN: opt < %s -passes='sample-profile,pgo-icall-prom' -sample-profile-file=%S/Inputs/vtable-type-prof.prof -enable-vtable-profile-use -icp-max-num-vtable-last-candidate=1 -pass-remarks=pgo-icall-prom -S 2>&1 | FileCheck %s --check-prefix=ICP

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@Base = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @Base_foo, ptr @Base_bar] }, !type !0
@Derived1 = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @Derived1_foo, ptr @Derived1_bar] }, !type !0, !type !1
@Derived2 = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @Base_foo, ptr @Derived2_bar] }, !type !0, !type !2

; ICP: remark: {{.*}} Promote indirect call to Derived1_bar with count 600 out of 1000, sink 2 instruction(s) and compare 1 vtable(s): {Derived1}
; ICP: remark: {{.*}} Promote indirect call to Derived2_bar with count 400 out of 400, sink 2 instruction(s) and compare 1 vtable(s): {Derived2}
; ICP: remark: {{.*}} Promote indirect call to Derived1_foo with count 1000 out of 1000, sink 2 instruction(s) and compare 1 vtable(s): {Derived1}

; COMMON-LABEL: define void @test(
; NOVTABLE: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}{{$}}
; VTABLE: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof ![[VT1:[0-9]+]]
; VTABLE1: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof ![[VT1:[0-9]+]]
; COMMON: call void %{{.*}}(ptr %d), !dbg !{{[0-9]+}}, !prof ![[FN1:[0-9]+]]
; NOVTABLE: %vtable2 = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}{{$}}
; VTABLE: %vtable2 = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof ![[VT2:[0-9]+]]
; VTABLE1: %vtable2 = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof ![[VT2:[0-9]+]]
; COMMON: call void %{{.*}}(ptr %d), !dbg !{{[0-9]+}}, !prof ![[FN2:[0-9]+]]
define void @test(ptr %d) #0 !dbg !10 {
entry:
  %vtable = load ptr, ptr %d, align 8, !dbg !11
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"Base"), !dbg !11
  tail call void @llvm.assume(i1 %0), !dbg !11
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1, !dbg !11
  %1 = load ptr, ptr %vfn, align 8, !dbg !11
  call void %1(ptr %d), !dbg !11
  %vtable2 = load ptr, ptr %d, align 8, !dbg !12
  %2 = tail call i1 @llvm.type.test(ptr %vtable2, metadata !"Base"), !dbg !12
  tail call void @llvm.assume(i1 %2), !dbg !12
  %vfn2 = getelementptr inbounds ptr, ptr %vtable2, i64 0, !dbg !12
  %3 = load ptr, ptr %vfn2, align 8, !dbg !12
  call void %3(ptr %d), !dbg !12
  ret void, !dbg !13
}

; A single vtable load (location 1) feeds two virtual calls; the type profile
; is keyed by the load's own location and annotated exactly once.
; COMMON-LABEL: define void @test_shared_vtable_load(
; NOVTABLE: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}{{$}}
; VTABLE: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof ![[VT2]]
; VTABLE-NOT: %vtable = load
; COMMON: call void %{{.*}}(ptr %d), !dbg !{{[0-9]+}}, !prof !{{[0-9]+}}
; COMMON: call void %{{.*}}(ptr %d), !dbg !{{[0-9]+}}, !prof !{{[0-9]+}}
define void @test_shared_vtable_load(ptr %d) #0 !dbg !20 {
entry:
  %vtable = load ptr, ptr %d, align 8, !dbg !21
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1, !dbg !21
  %0 = load ptr, ptr %vfn, align 8, !dbg !21
  call void %0(ptr %d), !dbg !21
  %vfn2 = getelementptr inbounds ptr, ptr %vtable, i64 0, !dbg !22
  %1 = load ptr, ptr %vfn2, align 8, !dbg !22
  call void %1(ptr %d), !dbg !22
  ret void, !dbg !23
}

; Existing (stale) vtable metadata on a cold virtual call is cleared when
; -overwrite-existing-weights is on, and otherwise left untouched.
; COMMON-LABEL: define void @test_cold(
; OVERWRITE: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}{{$}}
; NOVTABLE: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof !{{[0-9]+}}
; VTABLE1: %vtable = load ptr, ptr %d, align 8, !dbg !{{[0-9]+}}, !prof !{{[0-9]+}}
define void @test_cold(ptr %d, i1 %c) #0 !dbg !30 {
entry:
  br i1 %c, label %hot, label %cold, !dbg !33

hot:
  ret void, !dbg !32

cold:
  %vtable = load ptr, ptr %d, align 8, !dbg !31, !prof !40
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1, !dbg !31
  %0 = load ptr, ptr %vfn, align 8, !dbg !31
  call void %0(ptr %d), !dbg !31
  ret void, !dbg !32
}

define void @Base_foo(ptr %this) #0 {
  ret void
}

define void @Base_bar(ptr %this) #0 {
  ret void
}

define void @Derived1_foo(ptr %this) #0 {
  ret void
}

define void @Derived1_bar(ptr %this) #0 {
  ret void
}

define void @Derived2_bar(ptr %this) #0 {
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

attributes #0 = { "use-sample-profile" }

; Values are MD5 hashes of the vtable names (GUIDs), the same encoding ICP
; expects for IPVK_VTableTarget (kind 2).
; VTABLE-DAG: ![[VT1]] = !{!"VP", i32 2, i64 1000, i64 -4123858694673519054, i64 600, i64 -7211198353767973908, i64 400}
; VTABLE-DAG: ![[VT2]] = !{!"VP", i32 2, i64 1000, i64 -4123858694673519054, i64 1000}
; VTABLE1-DAG: ![[VT1]] = !{!"VP", i32 2, i64 1000, i64 -4123858694673519054, i64 600}
; VTABLE1-DAG: ![[VT2]] = !{!"VP", i32 2, i64 1000, i64 -4123858694673519054, i64 1000}
; COMMON-DAG: ![[FN1]] = !{!"VP", i32 0, i64 1000, i64 {{-?[0-9]+}}, i64 600, i64 {{-?[0-9]+}}, i64 400}
; COMMON-DAG: ![[FN2]] = !{!"VP", i32 0, i64 1000, i64 {{-?[0-9]+}}, i64 1000}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!4, !5}

!0 = !{i64 16, !"Base"}
!1 = !{i64 16, !"Derived1"}
!2 = !{i64 16, !"Derived2"}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !DIFile(filename: "test.cc", directory: "/")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!10 = distinct !DISubprogram(name: "test", scope: !6, file: !6, line: 10, type: !7, scopeLine: 10, unit: !3)
!11 = !DILocation(line: 11, column: 3, scope: !10)
!12 = !DILocation(line: 12, column: 3, scope: !10)
!13 = !DILocation(line: 13, column: 1, scope: !10)
!20 = distinct !DISubprogram(name: "test_shared_vtable_load", scope: !6, file: !6, line: 20, type: !7, scopeLine: 20, unit: !3)
!21 = !DILocation(line: 21, column: 3, scope: !20)
!22 = !DILocation(line: 23, column: 3, scope: !20)
!23 = !DILocation(line: 24, column: 1, scope: !20)
!30 = distinct !DISubprogram(name: "test_cold", scope: !6, file: !6, line: 30, type: !7, scopeLine: 30, unit: !3)
!31 = !DILocation(line: 31, column: 3, scope: !30)
!32 = !DILocation(line: 32, column: 1, scope: !30)
!33 = !DILocation(line: 33, column: 1, scope: !30)
!40 = !{!"VP", i32 2, i64 100, i64 -4123858694673519054, i64 100}
