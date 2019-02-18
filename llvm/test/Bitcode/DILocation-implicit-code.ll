; RUN: llvm-dis %s.bc -o - | FileCheck %s
; ModuleID = 'DILocation-implicit-code.cpp'
source_filename = "DILocation-implicit-code.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i8 }

$_ZN1AC2Ev = comdat any

$_ZN1A3fooEi = comdat any

@_ZTIi = external dso_local constant i8*

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z5test1v() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !7 {
entry:
  %retval = alloca %struct.A, align 1
  %a = alloca %struct.A, align 1
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %undef.agg.tmp = alloca %struct.A, align 1
  %e = alloca i32, align 4
  %undef.agg.tmp3 = alloca %struct.A, align 1
  call void @_ZN1AC2Ev(%struct.A* %a), !dbg !9
  invoke void @_ZN1A3fooEi(%struct.A* %a, i32 0)
          to label %invoke.cont unwind label %lpad, !dbg !10

invoke.cont:                                      ; preds = %entry
  br label %return, !dbg !11

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*), !dbg !12
  %1 = extractvalue { i8*, i32 } %0, 0, !dbg !12
  store i8* %1, i8** %exn.slot, align 8, !dbg !12
  %2 = extractvalue { i8*, i32 } %0, 1, !dbg !12
  store i32 %2, i32* %ehselector.slot, align 4, !dbg !12
  br label %catch.dispatch, !dbg !12

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot, align 4, !dbg !13
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #4, !dbg !13
  %matches = icmp eq i32 %sel, %3, !dbg !13
  br i1 %matches, label %catch, label %eh.resume, !dbg !13

catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot, align 8, !dbg !13
  %4 = call i8* @__cxa_begin_catch(i8* %exn) #4, !dbg !13
  %5 = bitcast i8* %4 to i32*, !dbg !13
  %6 = load i32, i32* %5, align 4, !dbg !13
  store i32 %6, i32* %e, align 4, !dbg !13
  %7 = load i32, i32* %e, align 4, !dbg !14
  invoke void @_ZN1A3fooEi(%struct.A* %a, i32 %7)
          to label %invoke.cont2 unwind label %lpad1, !dbg !15

invoke.cont2:                                     ; preds = %catch
  call void @__cxa_end_catch() #4, !dbg !16
  br label %return

lpad1:                                            ; preds = %catch
  %8 = landingpad { i8*, i32 }
          cleanup, !dbg !12
  %9 = extractvalue { i8*, i32 } %8, 0, !dbg !12
  store i8* %9, i8** %exn.slot, align 8, !dbg !12
  %10 = extractvalue { i8*, i32 } %8, 1, !dbg !12
  store i32 %10, i32* %ehselector.slot, align 4, !dbg !12
  call void @__cxa_end_catch() #4, !dbg !16
  br label %eh.resume, !dbg !16

try.cont:                                         ; No predecessors!
  call void @llvm.trap(), !dbg !16
  unreachable, !dbg !16

return:                                           ; preds = %invoke.cont2, %invoke.cont
  ret void, !dbg !12

eh.resume:                                        ; preds = %lpad1, %catch.dispatch
  %exn4 = load i8*, i8** %exn.slot, align 8, !dbg !13
  %sel5 = load i32, i32* %ehselector.slot, align 4, !dbg !13
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn4, 0, !dbg !13
  %lpad.val6 = insertvalue { i8*, i32 } %lpad.val, i32 %sel5, 1, !dbg !13
  resume { i8*, i32 } %lpad.val6, !dbg !13
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN1AC2Ev(%struct.A* %this) unnamed_addr #1 comdat align 2 !dbg !17 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  ret void, !dbg !18
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_ZN1A3fooEi(%struct.A* %this, i32 %i) #0 comdat align 2 !dbg !19 {
entry:
  %retval = alloca %struct.A, align 1
  %this.addr = alloca %struct.A*, align 8
  %i.addr = alloca i32, align 4
  store %struct.A* %this, %struct.A** %this.addr, align 8
  store i32 %i, i32* %i.addr, align 4
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  %0 = load i32, i32* %i.addr, align 4, !dbg !20
  %cmp = icmp eq i32 %0, 0, !dbg !21
  br i1 %cmp, label %if.then, label %if.end, !dbg !20

if.then:                                          ; preds = %entry
  %exception = call i8* @__cxa_allocate_exception(i64 4) #4, !dbg !22
  %1 = bitcast i8* %exception to i32*, !dbg !22
  store i32 1, i32* %1, align 16, !dbg !22
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #5, !dbg !22
  unreachable, !dbg !22

if.end:                                           ; preds = %entry
  ret void, !dbg !23
}

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

declare dso_local i8* @__cxa_begin_catch(i8*)

declare dso_local void @__cxa_end_catch()

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #3

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z5test2v() #0 !dbg !24 {
entry:
  %a = alloca %struct.A, align 1
  %b = alloca %struct.A, align 1
  %undef.agg.tmp = alloca %struct.A, align 1
  call void @_ZN1AC2Ev(%struct.A* %a), !dbg !25
  call void @_ZN1A3fooEi(%struct.A* %a, i32 1), !dbg !26
  ret void, !dbg !27
}

declare dso_local i8* @__cxa_allocate_exception(i64)

declare dso_local void @__cxa_throw(i8*, i8*, i8*)

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { noreturn nounwind }
attributes #4 = { nounwind }
attributes #5 = { noreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

; CHECK: !DILocation(line: 20, column: 1, scope: !{{[0-9]+}}, isImplicitCode: true)
; CHECK: !DILocation(line: 17, column: 5, scope: !{{[0-9]+}}, isImplicitCode: true)
; CHECK: !DILocation(line: 19, column: 5, scope: !{{[0-9]+}}, isImplicitCode: true)
; CHECK: !DILocation(line: 3, column: 10, scope: !{{[0-9]+}}, isImplicitCode: true)
; CHECK: !DILocation(line: 25, column: 1, scope: !{{[0-9]+}}, isImplicitCode: true)

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 342445)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "DILocation-implicit-code.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 342445)"}
!7 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 13, type: !8, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 14, column: 7, scope: !7)
!10 = !DILocation(line: 16, column: 18, scope: !7)
!11 = !DILocation(line: 16, column: 9, scope: !7)
!12 = !DILocation(line: 20, column: 1, scope: !7, isImplicitCode: true)
!13 = !DILocation(line: 17, column: 5, scope: !7, isImplicitCode: true)
!14 = !DILocation(line: 18, column: 22, scope: !7)
!15 = !DILocation(line: 18, column: 18, scope: !7)
!16 = !DILocation(line: 19, column: 5, scope: !7, isImplicitCode: true)
!17 = distinct !DISubprogram(name: "A", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 3, column: 10, scope: !17, isImplicitCode: true)
!19 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 5, column: 13, scope: !19)
!21 = !DILocation(line: 5, column: 15, scope: !19)
!22 = !DILocation(line: 6, column: 13, scope: !19)
!23 = !DILocation(line: 9, column: 9, scope: !19)
!24 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 22, type: !8, isLocal: false, isDefinition: true, scopeLine: 22, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!25 = !DILocation(line: 23, column: 7, scope: !24)
!26 = !DILocation(line: 24, column: 13, scope: !24)
!27 = !DILocation(line: 25, column: 1, scope: !24, isImplicitCode: true)
