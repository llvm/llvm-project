; RUN: opt -inline -S < %s | FileCheck %s

; Created from source
;
;
;  1 // test.cpp
;  2 extern int global_var;
;  3 extern int test_ext(int k);
;  4 int test (int k) {
;  5   int k2 = test_ext(k);
;  6   if (k2 > 100)
;  7     return k2;
;  8   return 0;
;  9 }
; 10
; 11 int test2() {
; 12   try
; 13   {
; 14     test(global_var);
; 15   }
; 16   catch (int e) {
; 17     global_var = 0;
; 18   }
; 19   global_var = 1;
; 20   return 0;
; 21 }

; CHECK: _Z4testi.exit:
; Make sure the branch instruction created during inlining has a debug location,
; so the range of the inlined function is correct.
; CHECK: br label %invoke.cont, !dbg ![[MD:[0-9]+]]
; The branch instruction has the source location of line 9 and its inlined location
; has the source location of line 14.
; CHECK: ![[INL:[0-9]+]] = metadata !{i32 14, i32 0, metadata {{.*}}, null}
; CHECK: ![[MD]] = metadata !{i32 9, i32 0, metadata {{.*}}, metadata ![[INL]]}

; ModuleID = 'test.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin12.0.0"

@_ZTIi = external constant i8*
@global_var = external global i32

define i32 @_Z4testi(i32 %k)  {
entry:
  %retval = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %k2 = alloca i32, align 4
  store i32 %k, i32* %k.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %k.addr}, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata !{i32* %k2}, metadata !15), !dbg !16
  %0 = load i32* %k.addr, align 4, !dbg !16
  %call = call i32 @_Z8test_exti(i32 %0), !dbg !16
  store i32 %call, i32* %k2, align 4, !dbg !16
  %1 = load i32* %k2, align 4, !dbg !17
  %cmp = icmp sgt i32 %1, 100, !dbg !17
  br i1 %cmp, label %if.then, label %if.end, !dbg !17

if.then:                                          ; preds = %entry
  %2 = load i32* %k2, align 4, !dbg !18
  store i32 %2, i32* %retval, !dbg !18
  br label %return, !dbg !18

if.end:                                           ; preds = %entry
  store i32 0, i32* %retval, !dbg !19
  br label %return, !dbg !19

return:                                           ; preds = %if.end, %if.then
  %3 = load i32* %retval, !dbg !20
  ret i32 %3, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare i32 @_Z8test_exti(i32)

define i32 @_Z5test2v()  {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  %0 = load i32* @global_var, align 4, !dbg !21
  %call = invoke i32 @_Z4testi(i32 %0)
          to label %invoke.cont unwind label %lpad, !dbg !21

invoke.cont:                                      ; preds = %entry
  br label %try.cont, !dbg !23

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*), !dbg !21
  %2 = extractvalue { i8*, i32 } %1, 0, !dbg !21
  store i8* %2, i8** %exn.slot, !dbg !21
  %3 = extractvalue { i8*, i32 } %1, 1, !dbg !21
  store i32 %3, i32* %ehselector.slot, !dbg !21
  br label %catch.dispatch, !dbg !21

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32* %ehselector.slot, !dbg !23
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2, !dbg !23
  %matches = icmp eq i32 %sel, %4, !dbg !23
  br i1 %matches, label %catch, label %eh.resume, !dbg !23

catch:                                            ; preds = %catch.dispatch
  call void @llvm.dbg.declare(metadata !{i32* %e}, metadata !24), !dbg !25
  %exn = load i8** %exn.slot, !dbg !23
  %5 = call i8* @__cxa_begin_catch(i8* %exn) #2, !dbg !23
  %6 = bitcast i8* %5 to i32*, !dbg !23
  %7 = load i32* %6, align 4, !dbg !23
  store i32 %7, i32* %e, align 4, !dbg !23
  store i32 0, i32* @global_var, align 4, !dbg !26
  call void @__cxa_end_catch() #2, !dbg !28
  br label %try.cont, !dbg !28

try.cont:                                         ; preds = %catch, %invoke.cont
  store i32 1, i32* @global_var, align 4, !dbg !29
  ret i32 0, !dbg !30

eh.resume:                                        ; preds = %catch.dispatch
  %exn1 = load i8** %exn.slot, !dbg !23
  %sel2 = load i32* %ehselector.slot, !dbg !23
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn1, 0, !dbg !23
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel2, 1, !dbg !23
  resume { i8*, i32 } %lpad.val3, !dbg !23
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!31}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !10}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"test", metadata !"test", metadata !"_Z4testi", i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z4testi, null, null, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [test]
!5 = metadata !{metadata !"test.cpp", metadata !""}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [test.cpp]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"test2", metadata !"test2", metadata !"_Z5test2v", i32 11, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z5test2v, null, null, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [test2]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !9}
!13 = metadata !{i32 786689, metadata !4, metadata !"k", metadata !6, i32 16777220, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [k] [line 4]
!14 = metadata !{i32 4, i32 0, metadata !4, null}
!15 = metadata !{i32 786688, metadata !4, metadata !"k2", metadata !6, i32 5, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [k2] [line 5]
!16 = metadata !{i32 5, i32 0, metadata !4, null}
!17 = metadata !{i32 6, i32 0, metadata !4, null}
!18 = metadata !{i32 7, i32 0, metadata !4, null}
!19 = metadata !{i32 8, i32 0, metadata !4, null}
!20 = metadata !{i32 9, i32 0, metadata !4, null}
!21 = metadata !{i32 14, i32 0, metadata !22, null}
!22 = metadata !{i32 786443, metadata !5, metadata !10, i32 13, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [test.cpp]
!23 = metadata !{i32 15, i32 0, metadata !22, null}
!24 = metadata !{i32 786688, metadata !10, metadata !"e", metadata !6, i32 16, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [e] [line 16]
!25 = metadata !{i32 16, i32 0, metadata !10, null}
!26 = metadata !{i32 17, i32 0, metadata !27, null}
!27 = metadata !{i32 786443, metadata !5, metadata !10, i32 16, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [test.cpp]
!28 = metadata !{i32 18, i32 0, metadata !27, null}
!29 = metadata !{i32 19, i32 0, metadata !10, null}
!30 = metadata !{i32 20, i32 0, metadata !10, null}
!31 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
