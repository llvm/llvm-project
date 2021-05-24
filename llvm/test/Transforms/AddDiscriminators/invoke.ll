; RUN: opt < %s -add-discriminators -S | FileCheck %s
; ModuleID = 'invoke.bc'
source_filename = "invoke.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Function Attrs: ssp uwtable
define void @_Z3foov() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !8 {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  ; CHECK: call void @_Z12bar_noexceptv({{.*}} !dbg ![[CALL1:[0-9]+]]
  call void @_Z12bar_noexceptv() #4, !dbg !11
  ; CHECK: call void @_Z12bar_noexceptv({{.*}} !dbg ![[CALL2:[0-9]+]]
  call void @_Z12bar_noexceptv() #4, !dbg !13
  invoke void @_Z3barv()
  ; CHECK: unwind label {{.*}} !dbg ![[INVOKE:[0-9]+]]
          to label %invoke.cont unwind label %lpad, !dbg !14

invoke.cont:                                      ; preds = %entry
  br label %try.cont, !dbg !15

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null, !dbg !16
  %1 = extractvalue { i8*, i32 } %0, 0, !dbg !16
  store i8* %1, i8** %exn.slot, align 8, !dbg !16
  %2 = extractvalue { i8*, i32 } %0, 1, !dbg !16
  store i32 %2, i32* %ehselector.slot, align 4, !dbg !16
  br label %catch, !dbg !16

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot, align 8, !dbg !15
  %3 = call i8* @__cxa_begin_catch(i8* %exn) #4, !dbg !15
  invoke void @__cxa_rethrow() #5
          to label %unreachable unwind label %lpad1, !dbg !17

lpad1:                                            ; preds = %catch
  %4 = landingpad { i8*, i32 }
          cleanup, !dbg !19
  %5 = extractvalue { i8*, i32 } %4, 0, !dbg !19
  store i8* %5, i8** %exn.slot, align 8, !dbg !19
  %6 = extractvalue { i8*, i32 } %4, 1, !dbg !19
  store i32 %6, i32* %ehselector.slot, align 4, !dbg !19
  invoke void @__cxa_end_catch()
          to label %invoke.cont2 unwind label %terminate.lpad, !dbg !20

invoke.cont2:                                     ; preds = %lpad1
  br label %eh.resume, !dbg !20

try.cont:                                         ; preds = %invoke.cont
  ret void, !dbg !21

eh.resume:                                        ; preds = %invoke.cont2
  %exn3 = load i8*, i8** %exn.slot, align 8, !dbg !20
  %sel = load i32, i32* %ehselector.slot, align 4, !dbg !20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn3, 0, !dbg !20
  %lpad.val4 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1, !dbg !20
  resume { i8*, i32 } %lpad.val4, !dbg !20

terminate.lpad:                                   ; preds = %lpad1
  %7 = landingpad { i8*, i32 }
          catch i8* null, !dbg !20
  %8 = extractvalue { i8*, i32 } %7, 0, !dbg !20
  call void @__clang_call_terminate(i8* %8) #6, !dbg !20
  unreachable, !dbg !20

unreachable:                                      ; preds = %catch
  unreachable
}

; Function Attrs: nounwind
declare void @_Z12bar_noexceptv() #1

declare void @_Z3barv() #2

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) #3 {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #4
  call void @_ZSt9terminatev() #6
  unreachable
}

declare void @_ZSt9terminatev()

attributes #0 = { ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline noreturn nounwind }
attributes #4 = { nounwind }
attributes #5 = { noreturn }
attributes #6 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

; CHECK: ![[CALL1]] = !DILocation(line: 7, column: 5, scope: ![[SCOPE1:[0-9]+]])
; CHECK: ![[SCOPE1]] = distinct !DILexicalBlock(scope: !8, file: !1, line: 6, column: 7)
; CHECK: ![[CALL2]] = !DILocation(line: 7, column: 21, scope: ![[SCOPE2:[0-9]+]])
; CHECK: ![[SCOPE2]] = !DILexicalBlockFile(scope: ![[SCOPE1]], file: !1, discriminator: 2)
; CHECK: ![[INVOKE]] = !DILocation(line: 7, column: 37, scope: ![[SCOPE3:[0-9]+]])
; CHECK: ![[SCOPE3]] = !DILexicalBlockFile(scope: ![[SCOPE1]], file: !1, discriminator: 4)

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: GNU)
!1 = !DIFile(filename: "invoke.cpp", directory: "examples")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 8.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 7, column: 5, scope: !12)
!12 = distinct !DILexicalBlock(scope: !8, file: !1, line: 6, column: 7)
!13 = !DILocation(line: 7, column: 21, scope: !12)
!14 = !DILocation(line: 7, column: 37, scope: !12)
!15 = !DILocation(line: 8, column: 3, scope: !12)
!16 = !DILocation(line: 12, column: 1, scope: !12)
!17 = !DILocation(line: 10, column: 5, scope: !18)
!18 = distinct !DILexicalBlock(scope: !8, file: !1, line: 9, column: 15)
!19 = !DILocation(line: 12, column: 1, scope: !18)
!20 = !DILocation(line: 11, column: 3, scope: !18)
!21 = !DILocation(line: 12, column: 1, scope: !8)
