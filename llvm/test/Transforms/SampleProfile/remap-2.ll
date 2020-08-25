; RUN: opt %s -passes=sample-profile -sample-profile-file=%S/Inputs/remap-2.prof -sample-profile-remapping-file=%S/Inputs/remap.map -S | FileCheck %s
; Check profile remapping works for searching inline instance, searching
; indirect call promotion candidate and prevent recursive inline.

@x.addr = common global i32 zeroinitializer, align 16
@y.addr = common global i32 zeroinitializer, align 16

define i32 @_ZN3foo3barERKN1M1XINS_6detail3quxEEE() #0 !dbg !9 {
entry:
  %t0 = load i32, i32* @x.addr, align 4
  %t1 = load i32, i32* @y.addr, align 4
  %add = add nsw i32 %t0, %t1
  ret i32 %add
}

define i32 @_ZN1M1XE() #0 !dbg !10 {
entry:
  %t0 = load i32, i32* @x.addr, align 4
  %t1 = load i32, i32* @y.addr, align 4
  %sub = sub nsw i32 %t0, %t1
  ret i32 %sub
}

define void @test(i32 ()*) #0 !dbg !4 {
  %t2 = alloca i32 ()*
  store i32 ()* %0, i32 ()** %t2
  %t3 = load i32 ()*, i32 ()** %t2
; Check call i32 %t3 has been indirect call promoted and call i32 @_ZN1M1XE
; has been inlined.
; CHECK-LABEL: @test(
; CHECK: icmp eq i32 ()* %t3, @_ZN3foo3barERKN1M1XINS_6detail3quxEEE
; CHECK-NOT: call i32 @_ZN1M1XE
  %t4 = call i32 %t3(), !dbg !7
  %t5 = call i32 @_ZN1M1XE(), !dbg !8
  ret void
}

define void @_ZN1M1X1YE(i32 ()*) #0 !dbg !11 {
  %t2 = alloca i32 ()*
  store i32 ()* %0, i32 ()** %t2
  %t3 = load i32 ()*, i32 ()** %t2
; Check call i32 %t3 has got its profile but is not indirect call promoted
; because the promotion candidate is a recursive call to the current function.
; CHECK-LABEL: @_ZN1M1X1YE(
; CHECK: call i32 %t3(), {{.*}} !prof ![[PROFID:[0-9]+]]
; CHECK-NOT: icmp eq i32 ()* %t3, @_ZN1M1X1YE
  %t4 = call i32 %t3(), !dbg !12
  ret void
}

; CHECK: ![[PROFID]] = !{!"VP", i32 0, i64 3000

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "calls.cc", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 8, scope: !4)
!8 = !DILocation(line: 9, scope: !4)
!9 = distinct !DISubprogram(name: "_ZN3foo3barERKN1M1XINS_6detail3quxEEE", line: 15, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!10 = distinct !DISubprogram(name: "_ZN1M1XE", line: 20, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!11 = distinct !DISubprogram(name: "_ZN1M1X1YE", line: 25, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!12 = !DILocation(line: 30, scope: !11)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.5 "}

