; RUN: opt -scalarrepl -disable-output < %s
; RUN: opt -scalarrepl-ssa -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; PR9017
define void @test1() nounwind readnone ssp {
entry:
  %l_72 = alloca i32*, align 8
  unreachable

for.cond:                                         ; preds = %for.cond
  %tmp1.i = load i32*, i32** %l_72, align 8
  store i32* %tmp1.i, i32** %l_72, align 8
  br label %for.cond

if.end:                                           ; No predecessors!
  ret void
}


define void @test2() {
  %E = alloca { { i32, float, double, i64 }, { i32, float, double, i64 } }        ; <{ { i32, float, double, i64 }, { i32, float, double, i64 } }*> [#uses=1]
  %tmp.151 = getelementptr { { i32, float, double, i64 }, { i32, float, double, i64 } }, { { i32, float, double, i64 }, { i32, float, double, i64 } }* %E, i64 0, i32 1, i32 3          ; <i64*> [#uses=0]
  ret void
}

define i32 @test3() {
        %X = alloca { [4 x i32] }               ; <{ [4 x i32] }*> [#uses=1]
        %Y = getelementptr { [4 x i32] }, { [4 x i32] }* %X, i64 0, i32 0, i64 2               ; <i32*> [#uses=2]
        store i32 4, i32* %Y
        %Z = load i32, i32* %Y               ; <i32> [#uses=1]
        ret i32 %Z
}


%struct.rtx_def = type { [2 x i8], i32, [1 x %union.rtunion_def] }
%union.rtunion_def = type { i32 }

define void @test4() {
entry:
        %c_addr.i = alloca i8           ; <i8*> [#uses=1]
        switch i32 0, label %return [
                 i32 36, label %label.7
                 i32 34, label %label.7
                 i32 41, label %label.5
        ]
label.5:                ; preds = %entry
        ret void
label.7:                ; preds = %entry, %entry
        br i1 false, label %then.4, label %switchexit.0
then.4:         ; preds = %label.7
        %tmp.0.i = bitcast i8* %c_addr.i to i32*                ; <i32*> [#uses=1]
        store i32 44, i32* %tmp.0.i
        ret void
switchexit.0:           ; preds = %label.7
        ret void
return:         ; preds = %entry
        ret void
}


define void @test5() {
entry:
        %source_ptr = alloca i8*, align 4               ; <i8**> [#uses=2]
        br i1 false, label %bb1357, label %cond_next583
cond_next583:           ; preds = %entry
        ret void
bb1357:         ; preds = %entry
        br i1 false, label %bb1365, label %bb27055
bb1365:         ; preds = %bb1357
        switch i32 0, label %cond_next10377 [
                 i32 0, label %bb4679
                 i32 1, label %bb4679
                 i32 2, label %bb4679
                 i32 3, label %bb4679
                 i32 4, label %bb5115
                 i32 5, label %bb6651
                 i32 6, label %bb7147
                 i32 7, label %bb8683
                 i32 8, label %bb9131
                 i32 9, label %bb9875
                 i32 10, label %bb4679
                 i32 11, label %bb4859
                 i32 12, label %bb4679
                 i32 16, label %bb10249
        ]
bb4679:         ; preds = %bb1365, %bb1365, %bb1365, %bb1365, %bb1365, %bb1365
        ret void
bb4859:         ; preds = %bb1365
        ret void
bb5115:         ; preds = %bb1365
        ret void
bb6651:         ; preds = %bb1365
        ret void
bb7147:         ; preds = %bb1365
        ret void
bb8683:         ; preds = %bb1365
        ret void
bb9131:         ; preds = %bb1365
        ret void
bb9875:         ; preds = %bb1365
        %source_ptr9884 = bitcast i8** %source_ptr to i8**              ; <i8**> [#uses=1]
        %tmp9885 = load i8*, i8** %source_ptr9884            ; <i8*> [#uses=0]
        ret void
bb10249:                ; preds = %bb1365
        %source_ptr10257 = bitcast i8** %source_ptr to i16**            ; <i16**> [#uses=1]
        %tmp10258 = load i16*, i16** %source_ptr10257         ; <i16*> [#uses=0]
        ret void
cond_next10377:         ; preds = %bb1365
        ret void
bb27055:                ; preds = %bb1357
        ret void
}


        %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>" = type { %"struct.__gnu_cxx::bitmap_allocator<char>::_Alloc_block"* }
        %"struct.__gnu_cxx::bitmap_allocator<char>" = type { i8 }
        %"struct.__gnu_cxx::bitmap_allocator<char>::_Alloc_block" = type { [8 x i8] }

; PR1045
define void @test6() {
entry:
        %this_addr.i = alloca %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"*                ; <%"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"**> [#uses=3]
        %tmp = alloca %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>", align 4                ; <%"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"*> [#uses=1]
        store %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"* %tmp, %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"** %this_addr.i
        %tmp.i = load %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"*, %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"** %this_addr.i          ; <%"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"*> [#uses=1]
        %tmp.i.upgrd.1 = bitcast %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"* %tmp.i to %"struct.__gnu_cxx::bitmap_allocator<char>"*              ; <%"struct.__gnu_cxx::bitmap_allocator<char>"*> [#uses=0]
        %tmp1.i = load %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"*, %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"** %this_addr.i         ; <%"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"*> [#uses=1]
        %tmp.i.upgrd.2 = getelementptr %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>", %"struct.__gnu_cxx::balloc::_Inclusive_between<__gnu_cxx::bitmap_allocator<char>::_Alloc_block*>"* %tmp1.i, i32 0, i32 0         ; <%"struct.__gnu_cxx::bitmap_allocator<char>::_Alloc_block"**> [#uses=0]
        unreachable
}

        %struct.CGPoint = type { float, float }
        %struct.aal_big_range_t = type { i32, i32 }        %struct.aal_callback_t = type { i8* (i8*, i32)*, void (i8*, i8*)* }        %struct.aal_edge_pool_t = type { %struct.aal_edge_pool_t*, i32, i32, [0 x %struct.aal_edge_t] }        %struct.aal_edge_t = type { %struct.CGPoint, %struct.CGPoint, i32 }
        %struct.aal_range_t = type { i16, i16 }
        %struct.aal_span_pool_t = type { %struct.aal_span_pool_t*, [341 x %struct.aal_span_t] }
        %struct.aal_span_t = type { %struct.aal_span_t*, %struct.aal_big_range_t }
        %struct.aal_spanarray_t = type { [2 x %struct.aal_range_t] }
        %struct.aal_spanbucket_t = type { i16, [2 x i8], %struct.anon }
        %struct.aal_state_t = type { %struct.CGPoint, %struct.CGPoint, %struct.CGPoint, i32, float, float, float, float, %struct.CGPoint, %struct.CGPoint, float, float, float, float, i32, i32, i32, i32, float, float, i8*, i32, i32, %struct.aal_edge_pool_t*, %struct.aal_edge_pool_t*, i8*, %struct.aal_callback_t*, i32, %struct.aal_span_t*, %struct.aal_span_t*, %struct.aal_span_t*, %struct.aal_span_pool_t*, i8, float, i8, i32 }
        %struct.anon = type { %struct.aal_spanarray_t }



define fastcc void @test7() {
entry:
        %SB = alloca %struct.aal_spanbucket_t, align 4          ; <%struct.aal_spanbucket_t*> [#uses=2]
        br i1 false, label %cond_true, label %cond_next79

cond_true:              ; preds = %entry
        br i1 false, label %cond_next, label %cond_next114.i

cond_next114.i:         ; preds = %cond_true
        ret void

cond_next:              ; preds = %cond_true
        %SB19 = bitcast %struct.aal_spanbucket_t* %SB to i8*            ; <i8*> [#uses=1]
        call void @llvm.memcpy.p0i8.p0i8.i32(i8* %SB19, i8* null, i32 12, i32 0, i1 false)
        br i1 false, label %cond_next34, label %cond_next79

cond_next34:            ; preds = %cond_next
        %i.2.reload22 = load i32, i32* null          ; <i32> [#uses=1]
        %tmp51 = getelementptr %struct.aal_spanbucket_t, %struct.aal_spanbucket_t* %SB, i32 0, i32 2, i32 0, i32 0, i32 %i.2.reload22, i32 1      
        ; <i16*> [#uses=0]
        ret void

cond_next79:            ; preds = %cond_next, %entry
        ret void
}


       %struct.c37304a__vrec = type { i8, %struct.c37304a__vrec___disc___XVN }
        %struct.c37304a__vrec___disc___XVN = type {
%struct.c37304a__vrec___disc___XVN___O }
        %struct.c37304a__vrec___disc___XVN___O = type {  }

; PR3304
define void @test8() {
entry:
        %v = alloca %struct.c37304a__vrec
        %0 = getelementptr %struct.c37304a__vrec, %struct.c37304a__vrec* %v, i32 0, i32 0             
        store i8 8, i8* %0, align 1
        unreachable
}



; rdar://6808691 - ZeroLengthMemSet
        %0 = type <{ i32, i16, i8, i8, i64, i64, i16, [0 x i16] }>           

define i32 @test9() {
entry:
        %.compoundliteral = alloca %0           
        %tmp228 = getelementptr %0, %0* %.compoundliteral, i32 0, i32 7
        %tmp229 = bitcast [0 x i16]* %tmp228 to i8*             
        call void @llvm.memset.p0i8.i64(i8* %tmp229, i8 0, i64 0, i32 2, i1 false)
        unreachable
}

declare void @llvm.memset.i64(i8* nocapture, i8, i64, i32) nounwind


; PR4146 - i1 handling
%wrapper = type { i1 }
define void @test10() {
entry:
        %w = alloca %wrapper, align 8           ; <%wrapper*> [#uses=1]
        %0 = getelementptr %wrapper, %wrapper* %w, i64 0, i32 0           ; <i1*>
        store i1 true, i1* %0
        ret void
}


        %struct.singlebool = type <{ i8 }>
; PR4286
define zeroext i8 @test11() nounwind {
entry:
        %a = alloca %struct.singlebool, align 1         ; <%struct.singlebool*> [#uses=2]
        %storetmp.i = bitcast %struct.singlebool* %a to i1*             ; <i1*> [#uses=1]
        store i1 true, i1* %storetmp.i
        %tmp = getelementptr %struct.singlebool, %struct.singlebool* %a, i64 0, i32 0               ; <i8*> [#uses=1]
        %tmp1 = load i8, i8* %tmp           ; <i8> [#uses=1]
        ret i8 %tmp1
}


       %struct.Item = type { [4 x i16], %struct.rule* }
        %struct.rule = type { [4 x i16], i32, i32, i32, %struct.nonterminal*, %struct.pattern*, i8 }
        %struct.nonterminal = type { i8*, i32, i32, i32, %struct.plankMap*, %struct.rule* }
        %struct.plankMap = type { %struct.list*, i32, %struct.stateMap* }
        %struct.list = type { i8*, %struct.list* }
        %struct.stateMap = type { i8*, %struct.plank*, i32, i16* }
        %struct.plank = type { i8*, %struct.list*, i32 }
        %struct.pattern = type { %struct.nonterminal*, %struct.operator*, [2 x %struct.nonterminal*] }
        %struct.operator = type { i8*, i8, i32, i32, i32, i32, %struct.table* }
        %struct.table = type { %struct.operator*, %struct.list*, i16*, [2 x %struct.dimension*], %struct.item_set** }
        %struct.dimension = type { i16*, %struct.Index_Map, %struct.mapping*, i32, %struct.plankMap* }
        %struct.Index_Map = type { i32, %struct.item_set** }
        %struct.item_set = type { i32, i32, %struct.operator*, [2 x %struct.item_set*], %struct.item_set*, i16*, %struct.Item*, %struct.Item* }
        %struct.mapping = type { %struct.list**, i32, i32, i32, %struct.item_set** }

; VLAs.
define void @test12() {
bb4.i:
        %malloccall = tail call i8* @malloc(i32 0)
        %0 = bitcast i8* %malloccall to [0 x %struct.Item]*
        %.sub.i.c.i = getelementptr [0 x %struct.Item], [0 x %struct.Item]* %0, i32 0, i32 0                ; <%struct.Item*> [#uses=0]
        unreachable
}
declare noalias i8* @malloc(i32)

; PR8680
define void @test13() nounwind {
entry:
  %memtmp = alloca i32, align 4
  %0 = bitcast i32* %memtmp to void ()*
  call void %0() nounwind
  ret void
}

; rdar://11861001 - The dynamic GEP here was incorrectly making all accesses
; to the alloca think they were also dynamic.  Inserts and extracts created to
; access the vector were all being based from the dynamic access, even in BBs
; not dominated by the GEP.
define fastcc void @test() optsize inlinehint ssp align 2 {
entry:
  %alloc.0.0 = alloca <4 x float>, align 16
  %bitcast = bitcast <4 x float>* %alloc.0.0 to [4 x float]*
  %idx3 = getelementptr inbounds [4 x float], [4 x float]* %bitcast, i32 0, i32 3
  store float 0.000000e+00, float* %idx3, align 4
  br label %for.body10

for.body10:                                       ; preds = %for.body10, %entry
  %loopidx = phi i32 [ 0, %entry ], [ undef, %for.body10 ]
  %unusedidx = getelementptr inbounds <4 x float>, <4 x float>* %alloc.0.0, i32 0, i32 %loopidx
  br i1 undef, label %for.end, label %for.body10

for.end:                                          ; preds = %for.body10
  store <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00>, <4 x float>* %alloc.0.0, align 16
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
