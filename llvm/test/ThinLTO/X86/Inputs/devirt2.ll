target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { ptr }
%struct.E = type { ptr }

@_ZTV1B = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1
@_ZTV1C = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1C1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !2
@_ZTV1D = linkonce_odr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1D1mEi] }, !type !3
@_ZTV1E = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1E1mEi] }, !type !4

define i32 @_ZN1B1fEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

define internal i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

define i32 @_ZN1C1fEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

define linkonce_odr i32 @_ZN1D1mEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

define internal i32 @_ZN1E1mEi(ptr %this, i32 %a) #0 {
   ret i32 0, !dbg !12
}

define i32 @test2(ptr %obj, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1E")
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  %call4 = tail call i32 %fptr33(ptr nonnull %obj, i32 %a)
  ret i32 %call4
}

attributes #0 = { noinline optnone }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!13, !14, !15}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}
!3 = !{i64 16, !"_ZTS1D"}
!4 = !{i64 16, !"_ZTS1E"}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !6, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, splitDebugInlining: false, nameTableKind: None)
!6 = !DIFile(filename: "test.cc", directory: "/tmp")
!7 = !{}
!8 = distinct !DISubprogram(name: "bar", linkageName: "_Z5barv", scope: !6, file: !6, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 2, column: 3, scope: !8, inlinedAt: !16)
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !DILocation(line: 1, column: 1, scope: !17)
!17 = distinct !DISubprogram(name: "foo", linkageName: "_Z5foov", scope: !6, file: !6, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
