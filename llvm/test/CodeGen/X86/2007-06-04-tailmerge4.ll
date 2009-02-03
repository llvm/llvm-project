; RUN: llvm-as < %s | llc -enable-eh | grep invcont131
; PR 1496:  tail merge was incorrectly removing this block

; ModuleID = 'report.1.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
  %struct.ALLOC = type { %struct.string___XUB, [2 x i8] }
  %struct.RETURN = type { i32, i32, i32, i64 }
  %struct.ada__streams__root_stream_type = type { %struct.ada__tags__dispatch_table* }
  %struct.ada__tags__dispatch_table = type { [1 x i8*] }
  %struct.ada__text_io__text_afcb = type { %struct.system__file_control_block__afcb, i32, i32, i32, i32, i32, %struct.ada__text_io__text_afcb*, i8, i8 }
  %struct.string___XUB = type { i32, i32 }
  %struct.string___XUP = type { i8*, %struct.string___XUB* }
  %struct.system__file_control_block__afcb = type { %struct.ada__streams__root_stream_type, i32, %struct.string___XUP, i32, %struct.string___XUP, i8, i8, i8, i8, i8, i8, i8, %struct.system__file_control_block__afcb*, %struct.system__file_control_block__afcb* }
  %struct.system__secondary_stack__mark_id = type { i8*, i32 }
  %struct.wide_string___XUP = type { i16*, %struct.string___XUB* }
@report_E = global i8 0   ; <i8*> [#uses=0]
@report__test_status = internal global i8 1   ; <i8*> [#uses=8]
@report__test_name = internal global [15 x i8] zeroinitializer    ; <[15 x i8]*> [#uses=10]
@report__test_name_len = internal global i32 0    ; <i32*> [#uses=15]
@.str = internal constant [12 x i8] c"report.adb\00\00"   ; <[12 x i8]*> [#uses=1]
@C.26.599 = internal constant %struct.string___XUB { i32 1, i32 1 }   ; <%struct.string___XUB*> [#uses=1]
@.str1 = internal constant [1 x i8] c":"    ; <[1 x i8]*> [#uses=1]
@.str2 = internal constant [1 x i8] c" "    ; <[1 x i8]*> [#uses=1]
@.str3 = internal constant [1 x i8] c"-"    ; <[1 x i8]*> [#uses=1]
@.str5 = internal constant [10 x i8] c"0123456789"    ; <[10 x i8]*> [#uses=12]
@C.59.855 = internal constant %struct.string___XUB { i32 1, i32 0 }   ; <%struct.string___XUB*> [#uses=1]
@C.69.876 = internal constant %struct.string___XUB { i32 1, i32 3 }   ; <%struct.string___XUB*> [#uses=1]
@C.70.879 = internal constant %struct.string___XUB { i32 1, i32 6 }   ; <%struct.string___XUB*> [#uses=1]
@C.81.900 = internal constant %struct.string___XUB { i32 1, i32 5 }   ; <%struct.string___XUB*> [#uses=1]
@.str6 = internal constant [0 x i8] zeroinitializer   ; <[0 x i8]*> [#uses=1]
@.str7 = internal constant [3 x i8] c"2.5"    ; <[3 x i8]*> [#uses=1]
@.str8 = internal constant [6 x i8] c"ACATS "   ; <[6 x i8]*> [#uses=1]
@.str9 = internal constant [5 x i8] c",.,. "    ; <[5 x i8]*> [#uses=1]
@.str10 = internal constant [1 x i8] c"."   ; <[1 x i8]*> [#uses=1]
@.str11 = internal constant [5 x i8] c"---- "   ; <[5 x i8]*> [#uses=1]
@.str12 = internal constant [5 x i8] c"   - "   ; <[5 x i8]*> [#uses=1]
@.str13 = internal constant [5 x i8] c"   * "   ; <[5 x i8]*> [#uses=1]
@.str14 = internal constant [5 x i8] c"   + "   ; <[5 x i8]*> [#uses=1]
@.str15 = internal constant [5 x i8] c"   ! "   ; <[5 x i8]*> [#uses=1]
@C.209.1380 = internal constant %struct.string___XUB { i32 1, i32 37 }    ; <%struct.string___XUB*> [#uses=1]
@.str16 = internal constant [37 x i8] c" PASSED ============================."    ; <[37 x i8]*> [#uses=1]
@.str17 = internal constant [5 x i8] c"==== "   ; <[5 x i8]*> [#uses=1]
@.str18 = internal constant [37 x i8] c" NOT-APPLICABLE ++++++++++++++++++++."    ; <[37 x i8]*> [#uses=1]
@.str19 = internal constant [5 x i8] c"++++ "   ; <[5 x i8]*> [#uses=1]
@.str20 = internal constant [37 x i8] c" TENTATIVELY PASSED !!!!!!!!!!!!!!!!."    ; <[37 x i8]*> [#uses=1]
@.str21 = internal constant [5 x i8] c"!!!! "   ; <[5 x i8]*> [#uses=1]
@.str22 = internal constant [37 x i8] c" SEE '!' COMMENTS FOR SPECIAL NOTES!!"    ; <[37 x i8]*> [#uses=1]
@.str23 = internal constant [37 x i8] c" FAILED ****************************."    ; <[37 x i8]*> [#uses=1]
@.str24 = internal constant [5 x i8] c"**** "   ; <[5 x i8]*> [#uses=1]
@__gnat_others_value = external constant i32    ; <i32*> [#uses=2]
@system__soft_links__abort_undefer = external global void ()*   ; <void ()**> [#uses=1]
@C.320.1854 = internal constant %struct.string___XUB { i32 2, i32 6 }   ; <%struct.string___XUB*> [#uses=1]

declare void @report__put_msg(i64 %msg.0.0)

declare void @__gnat_rcheck_05(i8*, i32)

declare void @__gnat_rcheck_12(i8*, i32)

declare %struct.ada__text_io__text_afcb* @ada__text_io__standard_output()

declare void @ada__text_io__set_col(%struct.ada__text_io__text_afcb*, i32)

declare void @ada__text_io__put_line(%struct.ada__text_io__text_afcb*, i64)

declare void @report__time_stamp(%struct.string___XUP* sret  %agg.result)

declare i64 @ada__calendar__clock()

declare void @ada__calendar__split(%struct.RETURN* sret , i64)

declare void @system__string_ops_concat_5__str_concat_5(%struct.string___XUP* sret , i64, i64, i64, i64, i64)

declare void @system__string_ops_concat_3__str_concat_3(%struct.string___XUP* sret , i64, i64, i64)

declare i8* @system__secondary_stack__ss_allocate(i32)

declare void @report__test(i64 %name.0.0, i64 %descr.0.0)

declare void @system__secondary_stack__ss_mark(%struct.system__secondary_stack__mark_id* sret )

declare i8* @llvm.eh.exception()

declare i32 @llvm.eh.selector(i8*, i8*, ...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)

declare void @__gnat_rcheck_07(i8*, i32)

declare void @system__secondary_stack__ss_release(i64)

declare void @report__comment(i64 %descr.0.0)

declare void @report__failed(i64 %descr.0.0)

declare void @report__not_applicable(i64 %descr.0.0)

declare void @report__special_action(i64 %descr.0.0)

define void @report__result() {
entry:
  %tmp = alloca %struct.system__secondary_stack__mark_id, align 8   ; <%struct.system__secondary_stack__mark_id*> [#uses=3]
  %A.210 = alloca %struct.string___XUB, align 8   ; <%struct.string___XUB*> [#uses=3]
  %tmp5 = alloca %struct.string___XUP, align 8    ; <%struct.string___XUP*> [#uses=3]
  %A.229 = alloca %struct.string___XUB, align 8   ; <%struct.string___XUB*> [#uses=3]
  %tmp10 = alloca %struct.string___XUP, align 8   ; <%struct.string___XUP*> [#uses=3]
  %A.248 = alloca %struct.string___XUB, align 8   ; <%struct.string___XUB*> [#uses=3]
  %tmp15 = alloca %struct.string___XUP, align 8   ; <%struct.string___XUP*> [#uses=3]
  %A.270 = alloca %struct.string___XUB, align 8   ; <%struct.string___XUB*> [#uses=3]
  %tmp20 = alloca %struct.string___XUP, align 8   ; <%struct.string___XUP*> [#uses=3]
  %A.284 = alloca %struct.string___XUB, align 8   ; <%struct.string___XUB*> [#uses=3]
  %tmp25 = alloca %struct.string___XUP, align 8   ; <%struct.string___XUP*> [#uses=3]
  call void @system__secondary_stack__ss_mark( %struct.system__secondary_stack__mark_id* %tmp sret  )
  %tmp28 = getelementptr %struct.system__secondary_stack__mark_id* %tmp, i32 0, i32 0   ; <i8**> [#uses=1]
  %tmp29 = load i8** %tmp28   ; <i8*> [#uses=2]
  %tmp31 = getelementptr %struct.system__secondary_stack__mark_id* %tmp, i32 0, i32 1   ; <i32*> [#uses=1]
  %tmp32 = load i32* %tmp31   ; <i32> [#uses=2]
  %tmp33 = load i8* @report__test_status    ; <i8> [#uses=1]
  switch i8 %tmp33, label %bb483 [
     i8 0, label %bb
     i8 2, label %bb143
     i8 3, label %bb261
  ]

bb:   ; preds = %entry
  %tmp34 = load i32* @report__test_name_len   ; <i32> [#uses=4]
  %tmp35 = icmp sgt i32 %tmp34, 0   ; <i1> [#uses=2]
  %tmp40 = icmp sgt i32 %tmp34, 15    ; <i1> [#uses=1]
  %bothcond139 = and i1 %tmp35, %tmp40    ; <i1> [#uses=1]
  br i1 %bothcond139, label %cond_true43, label %cond_next44

cond_true43:    ; preds = %bb
  invoke void @__gnat_rcheck_12( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 212 )
      to label %UnifiedUnreachableBlock unwind label %unwind

unwind:   ; preds = %invcont589, %cond_next567, %bb555, %cond_true497, %invcont249, %cond_next227, %bb215, %cond_true157, %invcont131, %cond_next109, %bb97, %cond_true43
  %eh_ptr = call i8* @llvm.eh.exception( )    ; <i8*> [#uses=1]
  br label %cleanup717

cond_next44:    ; preds = %bb
  %tmp72 = getelementptr %struct.string___XUB* %A.210, i32 0, i32 0   ; <i32*> [#uses=1]
  store i32 1, i32* %tmp72
  %tmp73 = getelementptr %struct.string___XUB* %A.210, i32 0, i32 1   ; <i32*> [#uses=1]
  store i32 %tmp34, i32* %tmp73
  br i1 %tmp35, label %cond_true80, label %cond_next109

cond_true80:    ; preds = %cond_next44
  %tmp45.off = add i32 %tmp34, -1   ; <i32> [#uses=1]
  %bothcond = icmp ugt i32 %tmp45.off, 14   ; <i1> [#uses=1]
  br i1 %bothcond, label %bb97, label %cond_next109

bb97:   ; preds = %cond_true80
  invoke void @__gnat_rcheck_05( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 212 )
      to label %UnifiedUnreachableBlock unwind label %unwind

cond_next109:   ; preds = %cond_true80, %cond_next44
  %A.210128 = ptrtoint %struct.string___XUB* %A.210 to i32    ; <i32> [#uses=1]
  %A.210128129 = zext i32 %A.210128 to i64    ; <i64> [#uses=1]
  %A.210128129130 = shl i64 %A.210128129, 32    ; <i64> [#uses=1]
  %A.210128129130.ins = or i64 %A.210128129130, zext (i32 ptrtoint ([15 x i8]* @report__test_name to i32) to i64)   ; <i64> [#uses=1]
  invoke void @system__string_ops_concat_3__str_concat_3( %struct.string___XUP* %tmp5 sret , i64 or (i64 zext (i32 ptrtoint ([5 x i8]* @.str17 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.81.900 to i32) to i64), i64 32)), i64 %A.210128129130.ins, i64 or (i64 zext (i32 ptrtoint ([37 x i8]* @.str16 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.209.1380 to i32) to i64), i64 32)) )
      to label %invcont131 unwind label %unwind

invcont131:   ; preds = %cond_next109
  %tmp133 = getelementptr %struct.string___XUP* %tmp5, i32 0, i32 0   ; <i8**> [#uses=1]
  %tmp134 = load i8** %tmp133   ; <i8*> [#uses=1]
  %tmp134120 = ptrtoint i8* %tmp134 to i32    ; <i32> [#uses=1]
  %tmp134120121 = zext i32 %tmp134120 to i64    ; <i64> [#uses=1]
  %tmp136 = getelementptr %struct.string___XUP* %tmp5, i32 0, i32 1   ; <%struct.string___XUB**> [#uses=1]
  %tmp137 = load %struct.string___XUB** %tmp136   ; <%struct.string___XUB*> [#uses=1]
  %tmp137116 = ptrtoint %struct.string___XUB* %tmp137 to i32    ; <i32> [#uses=1]
  %tmp137116117 = zext i32 %tmp137116 to i64    ; <i64> [#uses=1]
  %tmp137116117118 = shl i64 %tmp137116117, 32    ; <i64> [#uses=1]
  %tmp137116117118.ins = or i64 %tmp137116117118, %tmp134120121   ; <i64> [#uses=1]
  invoke fastcc void @report__put_msg( i64 %tmp137116117118.ins )
      to label %cond_next618 unwind label %unwind

bb143:    ; preds = %entry
  %tmp144 = load i32* @report__test_name_len    ; <i32> [#uses=4]
  %tmp147 = icmp sgt i32 %tmp144, 0   ; <i1> [#uses=2]
  %tmp154 = icmp sgt i32 %tmp144, 15    ; <i1> [#uses=1]
  %bothcond140 = and i1 %tmp147, %tmp154    ; <i1> [#uses=1]
  br i1 %bothcond140, label %cond_true157, label %cond_next160

cond_true157:   ; preds = %bb143
  invoke void @__gnat_rcheck_12( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 215 )
      to label %UnifiedUnreachableBlock unwind label %unwind

cond_next160:   ; preds = %bb143
  %tmp189 = getelementptr %struct.string___XUB* %A.229, i32 0, i32 0    ; <i32*> [#uses=1]
  store i32 1, i32* %tmp189
  %tmp190 = getelementptr %struct.string___XUB* %A.229, i32 0, i32 1    ; <i32*> [#uses=1]
  store i32 %tmp144, i32* %tmp190
  br i1 %tmp147, label %cond_true197, label %cond_next227

cond_true197:   ; preds = %cond_next160
  %tmp161.off = add i32 %tmp144, -1   ; <i32> [#uses=1]
  %bothcond1 = icmp ugt i32 %tmp161.off, 14   ; <i1> [#uses=1]
  br i1 %bothcond1, label %bb215, label %cond_next227

bb215:    ; preds = %cond_true197
  invoke void @__gnat_rcheck_05( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 215 )
      to label %UnifiedUnreachableBlock unwind label %unwind

cond_next227:   ; preds = %cond_true197, %cond_next160
  %A.229105 = ptrtoint %struct.string___XUB* %A.229 to i32    ; <i32> [#uses=1]
  %A.229105106 = zext i32 %A.229105 to i64    ; <i64> [#uses=1]
  %A.229105106107 = shl i64 %A.229105106, 32    ; <i64> [#uses=1]
  %A.229105106107.ins = or i64 %A.229105106107, zext (i32 ptrtoint ([15 x i8]* @report__test_name to i32) to i64)   ; <i64> [#uses=1]
  invoke void @system__string_ops_concat_3__str_concat_3( %struct.string___XUP* %tmp10 sret , i64 or (i64 zext (i32 ptrtoint ([5 x i8]* @.str19 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.81.900 to i32) to i64), i64 32)), i64 %A.229105106107.ins, i64 or (i64 zext (i32 ptrtoint ([37 x i8]* @.str18 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.209.1380 to i32) to i64), i64 32)) )
      to label %invcont249 unwind label %unwind

invcont249:   ; preds = %cond_next227
  %tmp251 = getelementptr %struct.string___XUP* %tmp10, i32 0, i32 0    ; <i8**> [#uses=1]
  %tmp252 = load i8** %tmp251   ; <i8*> [#uses=1]
  %tmp25297 = ptrtoint i8* %tmp252 to i32   ; <i32> [#uses=1]
  %tmp2529798 = zext i32 %tmp25297 to i64   ; <i64> [#uses=1]
  %tmp254 = getelementptr %struct.string___XUP* %tmp10, i32 0, i32 1    ; <%struct.string___XUB**> [#uses=1]
  %tmp255 = load %struct.string___XUB** %tmp254   ; <%struct.string___XUB*> [#uses=1]
  %tmp25593 = ptrtoint %struct.string___XUB* %tmp255 to i32   ; <i32> [#uses=1]
  %tmp2559394 = zext i32 %tmp25593 to i64   ; <i64> [#uses=1]
  %tmp255939495 = shl i64 %tmp2559394, 32   ; <i64> [#uses=1]
  %tmp255939495.ins = or i64 %tmp255939495, %tmp2529798   ; <i64> [#uses=1]
  invoke fastcc void @report__put_msg( i64 %tmp255939495.ins )
      to label %cond_next618 unwind label %unwind

bb261:    ; preds = %entry
  %tmp262 = call i8* @llvm.stacksave( )   ; <i8*> [#uses=2]
  %tmp263 = load i32* @report__test_name_len    ; <i32> [#uses=4]
  %tmp266 = icmp sgt i32 %tmp263, 0   ; <i1> [#uses=2]
  %tmp273 = icmp sgt i32 %tmp263, 15    ; <i1> [#uses=1]
  %bothcond141 = and i1 %tmp266, %tmp273    ; <i1> [#uses=1]
  br i1 %bothcond141, label %cond_true276, label %cond_next281

cond_true276:   ; preds = %bb261
  invoke void @__gnat_rcheck_12( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 218 )
      to label %UnifiedUnreachableBlock unwind label %unwind277

unwind277:    ; preds = %invcont467, %cond_next442, %invcont370, %cond_next348, %bb336, %cond_true276
  %eh_ptr278 = call i8* @llvm.eh.exception( )   ; <i8*> [#uses=1]
  call void @llvm.stackrestore( i8* %tmp262 )
  br label %cleanup717

cond_next281:   ; preds = %bb261
  %tmp310 = getelementptr %struct.string___XUB* %A.248, i32 0, i32 0    ; <i32*> [#uses=1]
  store i32 1, i32* %tmp310
  %tmp311 = getelementptr %struct.string___XUB* %A.248, i32 0, i32 1    ; <i32*> [#uses=1]
  store i32 %tmp263, i32* %tmp311
  br i1 %tmp266, label %cond_true318, label %cond_next348

cond_true318:   ; preds = %cond_next281
  %tmp282.off = add i32 %tmp263, -1   ; <i32> [#uses=1]
  %bothcond2 = icmp ugt i32 %tmp282.off, 14   ; <i1> [#uses=1]
  br i1 %bothcond2, label %bb336, label %cond_next348

bb336:    ; preds = %cond_true318
  invoke void @__gnat_rcheck_05( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 218 )
      to label %UnifiedUnreachableBlock unwind label %unwind277

cond_next348:   ; preds = %cond_true318, %cond_next281
  %A.24882 = ptrtoint %struct.string___XUB* %A.248 to i32   ; <i32> [#uses=1]
  %A.2488283 = zext i32 %A.24882 to i64   ; <i64> [#uses=1]
  %A.248828384 = shl i64 %A.2488283, 32   ; <i64> [#uses=1]
  %A.248828384.ins = or i64 %A.248828384, zext (i32 ptrtoint ([15 x i8]* @report__test_name to i32) to i64)   ; <i64> [#uses=1]
  invoke void @system__string_ops_concat_3__str_concat_3( %struct.string___XUP* %tmp15 sret , i64 or (i64 zext (i32 ptrtoint ([5 x i8]* @.str21 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.81.900 to i32) to i64), i64 32)), i64 %A.248828384.ins, i64 or (i64 zext (i32 ptrtoint ([37 x i8]* @.str20 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.209.1380 to i32) to i64), i64 32)) )
      to label %invcont370 unwind label %unwind277

invcont370:   ; preds = %cond_next348
  %tmp372 = getelementptr %struct.string___XUP* %tmp15, i32 0, i32 0    ; <i8**> [#uses=1]
  %tmp373 = load i8** %tmp372   ; <i8*> [#uses=1]
  %tmp37374 = ptrtoint i8* %tmp373 to i32   ; <i32> [#uses=1]
  %tmp3737475 = zext i32 %tmp37374 to i64   ; <i64> [#uses=1]
  %tmp375 = getelementptr %struct.string___XUP* %tmp15, i32 0, i32 1    ; <%struct.string___XUB**> [#uses=1]
  %tmp376 = load %struct.string___XUB** %tmp375   ; <%struct.string___XUB*> [#uses=1]
  %tmp37670 = ptrtoint %struct.string___XUB* %tmp376 to i32   ; <i32> [#uses=1]
  %tmp3767071 = zext i32 %tmp37670 to i64   ; <i64> [#uses=1]
  %tmp376707172 = shl i64 %tmp3767071, 32   ; <i64> [#uses=1]
  %tmp376707172.ins = or i64 %tmp376707172, %tmp3737475   ; <i64> [#uses=1]
  invoke fastcc void @report__put_msg( i64 %tmp376707172.ins )
      to label %invcont381 unwind label %unwind277

invcont381:   ; preds = %invcont370
  %tmp382 = load i32* @report__test_name_len    ; <i32> [#uses=6]
  %tmp415 = icmp sgt i32 %tmp382, -1    ; <i1> [#uses=1]
  %max416 = select i1 %tmp415, i32 %tmp382, i32 0   ; <i32> [#uses=1]
  %tmp417 = alloca i8, i32 %max416    ; <i8*> [#uses=3]
  %tmp423 = icmp sgt i32 %tmp382, 0   ; <i1> [#uses=1]
  br i1 %tmp423, label %bb427, label %cond_next442

bb427:    ; preds = %invcont381
  store i8 32, i8* %tmp417
  %tmp434 = icmp eq i32 %tmp382, 1    ; <i1> [#uses=1]
  br i1 %tmp434, label %cond_next442, label %cond_next438.preheader

cond_next438.preheader:   ; preds = %bb427
  %tmp. = add i32 %tmp382, -1   ; <i32> [#uses=1]
  br label %cond_next438

cond_next438:   ; preds = %cond_next438, %cond_next438.preheader
  %indvar = phi i32 [ 0, %cond_next438.preheader ], [ %J130b.513.5, %cond_next438 ]   ; <i32> [#uses=1]
  %J130b.513.5 = add i32 %indvar, 1   ; <i32> [#uses=3]
  %tmp43118 = getelementptr i8* %tmp417, i32 %J130b.513.5   ; <i8*> [#uses=1]
  store i8 32, i8* %tmp43118
  %exitcond = icmp eq i32 %J130b.513.5, %tmp.   ; <i1> [#uses=1]
  br i1 %exitcond, label %cond_next442, label %cond_next438

cond_next442:   ; preds = %cond_next438, %bb427, %invcont381
  %tmp448 = getelementptr %struct.string___XUB* %A.270, i32 0, i32 0    ; <i32*> [#uses=1]
  store i32 1, i32* %tmp448
  %tmp449 = getelementptr %struct.string___XUB* %A.270, i32 0, i32 1    ; <i32*> [#uses=1]
  store i32 %tmp382, i32* %tmp449
  %tmp41762 = ptrtoint i8* %tmp417 to i32   ; <i32> [#uses=1]
  %tmp4176263 = zext i32 %tmp41762 to i64   ; <i64> [#uses=1]
  %A.27058 = ptrtoint %struct.string___XUB* %A.270 to i32   ; <i32> [#uses=1]
  %A.2705859 = zext i32 %A.27058 to i64   ; <i64> [#uses=1]
  %A.270585960 = shl i64 %A.2705859, 32   ; <i64> [#uses=1]
  %A.270585960.ins = or i64 %tmp4176263, %A.270585960   ; <i64> [#uses=1]
  invoke void @system__string_ops_concat_3__str_concat_3( %struct.string___XUP* %tmp20 sret , i64 or (i64 zext (i32 ptrtoint ([5 x i8]* @.str21 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.81.900 to i32) to i64), i64 32)), i64 %A.270585960.ins, i64 or (i64 zext (i32 ptrtoint ([37 x i8]* @.str22 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.209.1380 to i32) to i64), i64 32)) )
      to label %invcont467 unwind label %unwind277

invcont467:   ; preds = %cond_next442
  %tmp469 = getelementptr %struct.string___XUP* %tmp20, i32 0, i32 0    ; <i8**> [#uses=1]
  %tmp470 = load i8** %tmp469   ; <i8*> [#uses=1]
  %tmp47050 = ptrtoint i8* %tmp470 to i32   ; <i32> [#uses=1]
  %tmp4705051 = zext i32 %tmp47050 to i64   ; <i64> [#uses=1]
  %tmp472 = getelementptr %struct.string___XUP* %tmp20, i32 0, i32 1    ; <%struct.string___XUB**> [#uses=1]
  %tmp473 = load %struct.string___XUB** %tmp472   ; <%struct.string___XUB*> [#uses=1]
  %tmp47346 = ptrtoint %struct.string___XUB* %tmp473 to i32   ; <i32> [#uses=1]
  %tmp4734647 = zext i32 %tmp47346 to i64   ; <i64> [#uses=1]
  %tmp473464748 = shl i64 %tmp4734647, 32   ; <i64> [#uses=1]
  %tmp473464748.ins = or i64 %tmp473464748, %tmp4705051   ; <i64> [#uses=1]
  invoke fastcc void @report__put_msg( i64 %tmp473464748.ins )
      to label %cleanup unwind label %unwind277

cleanup:    ; preds = %invcont467
  call void @llvm.stackrestore( i8* %tmp262 )
  br label %cond_next618

bb483:    ; preds = %entry
  %tmp484 = load i32* @report__test_name_len    ; <i32> [#uses=4]
  %tmp487 = icmp sgt i32 %tmp484, 0   ; <i1> [#uses=2]
  %tmp494 = icmp sgt i32 %tmp484, 15    ; <i1> [#uses=1]
  %bothcond142 = and i1 %tmp487, %tmp494    ; <i1> [#uses=1]
  br i1 %bothcond142, label %cond_true497, label %cond_next500

cond_true497:   ; preds = %bb483
  invoke void @__gnat_rcheck_12( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 223 )
      to label %UnifiedUnreachableBlock unwind label %unwind

cond_next500:   ; preds = %bb483
  %tmp529 = getelementptr %struct.string___XUB* %A.284, i32 0, i32 0    ; <i32*> [#uses=1]
  store i32 1, i32* %tmp529
  %tmp530 = getelementptr %struct.string___XUB* %A.284, i32 0, i32 1    ; <i32*> [#uses=1]
  store i32 %tmp484, i32* %tmp530
  br i1 %tmp487, label %cond_true537, label %cond_next567

cond_true537:   ; preds = %cond_next500
  %tmp501.off = add i32 %tmp484, -1   ; <i32> [#uses=1]
  %bothcond3 = icmp ugt i32 %tmp501.off, 14   ; <i1> [#uses=1]
  br i1 %bothcond3, label %bb555, label %cond_next567

bb555:    ; preds = %cond_true537
  invoke void @__gnat_rcheck_05( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 223 )
      to label %UnifiedUnreachableBlock unwind label %unwind

cond_next567:   ; preds = %cond_true537, %cond_next500
  %A.28435 = ptrtoint %struct.string___XUB* %A.284 to i32   ; <i32> [#uses=1]
  %A.2843536 = zext i32 %A.28435 to i64   ; <i64> [#uses=1]
  %A.284353637 = shl i64 %A.2843536, 32   ; <i64> [#uses=1]
  %A.284353637.ins = or i64 %A.284353637, zext (i32 ptrtoint ([15 x i8]* @report__test_name to i32) to i64)   ; <i64> [#uses=1]
  invoke void @system__string_ops_concat_3__str_concat_3( %struct.string___XUP* %tmp25 sret , i64 or (i64 zext (i32 ptrtoint ([5 x i8]* @.str24 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.81.900 to i32) to i64), i64 32)), i64 %A.284353637.ins, i64 or (i64 zext (i32 ptrtoint ([37 x i8]* @.str23 to i32) to i64), i64 shl (i64 zext (i32 ptrtoint (%struct.string___XUB* @C.209.1380 to i32) to i64), i64 32)) )
      to label %invcont589 unwind label %unwind

invcont589:   ; preds = %cond_next567
  %tmp591 = getelementptr %struct.string___XUP* %tmp25, i32 0, i32 0    ; <i8**> [#uses=1]
  %tmp592 = load i8** %tmp591   ; <i8*> [#uses=1]
  %tmp59228 = ptrtoint i8* %tmp592 to i32   ; <i32> [#uses=1]
  %tmp5922829 = zext i32 %tmp59228 to i64   ; <i64> [#uses=1]
  %tmp594 = getelementptr %struct.string___XUP* %tmp25, i32 0, i32 1    ; <%struct.string___XUB**> [#uses=1]
  %tmp595 = load %struct.string___XUB** %tmp594   ; <%struct.string___XUB*> [#uses=1]
  %tmp59524 = ptrtoint %struct.string___XUB* %tmp595 to i32   ; <i32> [#uses=1]
  %tmp5952425 = zext i32 %tmp59524 to i64   ; <i64> [#uses=1]
  %tmp595242526 = shl i64 %tmp5952425, 32   ; <i64> [#uses=1]
  %tmp595242526.ins = or i64 %tmp595242526, %tmp5922829   ; <i64> [#uses=1]
  invoke fastcc void @report__put_msg( i64 %tmp595242526.ins )
      to label %cond_next618 unwind label %unwind

cond_next618:   ; preds = %invcont589, %cleanup, %invcont249, %invcont131
  store i8 1, i8* @report__test_status
  store i32 7, i32* @report__test_name_len
  store i8 78, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 0)
  store i8 79, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 1)
  store i8 95, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 2)
  store i8 78, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 3)
  store i8 65, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 4)
  store i8 77, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 5)
  store i8 69, i8* getelementptr ([15 x i8]* @report__test_name, i32 0, i32 6)
  %CHAIN.310.0.0.0.val5.i = ptrtoint i8* %tmp29 to i32    ; <i32> [#uses=1]
  %CHAIN.310.0.0.0.val56.i = zext i32 %CHAIN.310.0.0.0.val5.i to i64    ; <i64> [#uses=1]
  %CHAIN.310.0.0.1.val2.i = zext i32 %tmp32 to i64    ; <i64> [#uses=1]
  %CHAIN.310.0.0.1.val23.i = shl i64 %CHAIN.310.0.0.1.val2.i, 32    ; <i64> [#uses=1]
  %CHAIN.310.0.0.1.val23.ins.i = or i64 %CHAIN.310.0.0.1.val23.i, %CHAIN.310.0.0.0.val56.i    ; <i64> [#uses=1]
  call void @system__secondary_stack__ss_release( i64 %CHAIN.310.0.0.1.val23.ins.i )
  ret void

cleanup717:   ; preds = %unwind277, %unwind
  %eh_exception.0 = phi i8* [ %eh_ptr278, %unwind277 ], [ %eh_ptr, %unwind ]    ; <i8*> [#uses=1]
  %CHAIN.310.0.0.0.val5.i8 = ptrtoint i8* %tmp29 to i32   ; <i32> [#uses=1]
  %CHAIN.310.0.0.0.val56.i9 = zext i32 %CHAIN.310.0.0.0.val5.i8 to i64    ; <i64> [#uses=1]
  %CHAIN.310.0.0.1.val2.i10 = zext i32 %tmp32 to i64    ; <i64> [#uses=1]
  %CHAIN.310.0.0.1.val23.i11 = shl i64 %CHAIN.310.0.0.1.val2.i10, 32    ; <i64> [#uses=1]
  %CHAIN.310.0.0.1.val23.ins.i12 = or i64 %CHAIN.310.0.0.1.val23.i11, %CHAIN.310.0.0.0.val56.i9   ; <i64> [#uses=1]
  call void @system__secondary_stack__ss_release( i64 %CHAIN.310.0.0.1.val23.ins.i12 )
  call i32 (...)* @_Unwind_Resume( i8* %eh_exception.0 )    ; <i32>:0 [#uses=0]
  unreachable

UnifiedUnreachableBlock:    ; preds = %bb555, %cond_true497, %bb336, %cond_true276, %bb215, %cond_true157, %bb97, %cond_true43
  unreachable
}

declare i8* @llvm.stacksave()

declare void @llvm.stackrestore(i8*)

declare i32 @report__ident_int(i32 %x)

declare i8 @report__equal(i32 %x, i32 %y)

declare i8 @report__ident_char(i8 zeroext  %x)

declare i16 @report__ident_wide_char(i16 zeroext  %x)

declare i8 @report__ident_bool(i8 %x)

declare void @report__ident_str(%struct.string___XUP* sret  %agg.result, i64 %x.0.0)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @report__ident_wide_str(%struct.wide_string___XUP* sret  %agg.result, i64 %x.0.0)

declare void @__gnat_begin_handler(i8*)

declare void @__gnat_end_handler(i8*)

declare void @report__legal_file_name(%struct.string___XUP* sret  %agg.result, i32 %x, i64 %nam.0.0)

declare void @__gnat_rcheck_06(i8*, i32)

declare void @system__string_ops__str_concat_cs(%struct.string___XUP* sret , i8 zeroext , i64)
