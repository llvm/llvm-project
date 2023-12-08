; Test the profile _Z3sumii won't be mistakenly annotated to foo
; when '-sample-profile-merge-inlinee' is enabled. When the mistake
; happens, there will be a "No debug information found" warning message.
; RUN: opt < %s -passes=sample-profile \
; RUN:   -sample-profile-file=%S/Inputs/inline-mergeprof.prof \
; RUN:   -sample-profile-merge-inlinee=true -S 2>&1| FileCheck %s

@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1
declare void @__cxa_call_unexpected(ptr)
declare i32 @__gxx_personality_v0(...)
declare i32 @_Z3subii(i32 %x, i32 %y)

define i32 @main() "use-sample-profile" nounwind uwtable ssp personality ptr @__gxx_personality_v0 !dbg !6 {
entry:
  %retval = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  %tmp = load i32, ptr %i, align 4, !dbg !8
  %tmp1 = load i32, ptr %s, align 4, !dbg !8
  %call = invoke i32 @foo(i32 %tmp, i32 %tmp1)
          to label %cont unwind label %lpad, !dbg !8
; CHECK-NOT: warning: No debug information found in function foo
; CHECK: invoke i32 @foo
cont:
  store i32 %call, ptr %s, align 4, !dbg !8
  ret i32 0, !dbg !11
lpad:
  %lptmp0 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %lptmp1 = extractvalue { ptr, i32 } %lptmp0, 0
  tail call void @__cxa_call_unexpected(ptr %lptmp1) noreturn nounwind
  unreachable
}

define i32 @foo(i32 %x, i32 %y) #0 {
entry:
  %add = add nsw i32 %x, %y
  ret i32 %add
}

attributes #0 = { "use-sample-profile" }

declare i32 @printf(ptr, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.5 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !7, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 10, scope: !9)
!9 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 2)
!10 = distinct !DILexicalBlock(scope: !6, file: !1, line: 10)
!11 = !DILocation(line: 12, scope: !6)
