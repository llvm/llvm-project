; RUN: llc < %s -mtriple=x86_64-apple-macosx -regalloc=greedy | FileCheck %s

; This testing case is reduced from 197.parser prune_match function.
; We make sure register copies are not generated on isupper.exit blocks.

; isupper.exit and isupper.exit223 get tail-duplicated into all their
; predecessors.
; CHECK: cond.true.i.i
; CHECK-NEXT: in Loop
; Mem-move
; CHECK-NEXT: movl
; CHECK-NEXT: andl
; CHECK-NEXT: testl
; CHECK-NEXT: jne
; CHECK: cond.true.i.i217
; CHECK-NEXT: in Loop
; Mem-move
; CHECK-NEXT: movl
; CHECK-NEXT: andl
; CHECK-NEXT: testl
; CHECK-NEXT: je
; CHECK: cond.false.i.i
; CHECK: maskrune
; CHECK-NEXT: movzbl
; CHECK-NEXT: movzbl
; CHECK-NEXT: testl
; CHECK-NEXT: je
; CHECK: cond.false.i.i219
; CHECK: maskrune
; CHECK-NEXT: movzbl
; CHECK-NEXT: movzbl
; CHECK-NEXT: testl
; CHECK-NEXT: jne

%struct.List_o_links_struct = type { i32, i32, i32, %struct.List_o_links_struct* }
%struct.Connector_struct = type { i16, i16, i8, i8, %struct.Connector_struct*, i8* }
%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i64, i8**)*, i32 (i32, i8*, i64, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32, i32, %struct._RuneCharClass* }
%struct._RuneRange = type { i32, %struct._RuneEntry* }
%struct._RuneEntry = type { i32, i32, i32, i32* }
%struct._RuneCharClass = type { [14 x i8], i32 }
%struct.Exp_struct = type { i8, i8, i8, i8, %union.anon }
%union.anon = type { %struct.E_list_struct* }
%struct.E_list_struct = type { %struct.E_list_struct*, %struct.Exp_struct* }
%struct.domain_struct = type { i8*, i32, %struct.List_o_links_struct*, i32, i32, %struct.d_tree_leaf_struct*, %struct.domain_struct* }
%struct.d_tree_leaf_struct = type { %struct.domain_struct*, i32, %struct.d_tree_leaf_struct* }
@_DefaultRuneLocale = external global %struct._RuneLocale
declare i32 @__maskrune(i32, i64) #7
define fastcc i32 @prune_match(%struct.Connector_struct* nocapture readonly %a, %struct.Connector_struct* nocapture readonly %b) #9 {
entry:
  %label56 = bitcast %struct.Connector_struct* %a to i16*
  %0 = load i16, i16* %label56, align 2
  %label157 = bitcast %struct.Connector_struct* %b to i16*
  %1 = load i16, i16* %label157, align 2
  %cmp = icmp eq i16 %0, %1
  br i1 %cmp, label %if.end, label %return, !prof !988
if.end:
  %priority = getelementptr inbounds %struct.Connector_struct, %struct.Connector_struct* %a, i64 0, i32 2
  %2 = load i8, i8* %priority, align 1
  %priority5 = getelementptr inbounds %struct.Connector_struct, %struct.Connector_struct* %b, i64 0, i32 2
  %3 = load i8, i8* %priority5, align 1
  %string = getelementptr inbounds %struct.Connector_struct, %struct.Connector_struct* %a, i64 0, i32 5
  %4 = load i8*, i8** %string, align 8
  %string7 = getelementptr inbounds %struct.Connector_struct, %struct.Connector_struct* %b, i64 0, i32 5
  %5 = load i8*, i8** %string7, align 8
  br label %while.cond
while.cond:
  %lsr.iv27 = phi i64 [ %lsr.iv.next28, %if.end17 ], [ 0, %if.end ]
  %scevgep55 = getelementptr i8, i8* %4, i64 %lsr.iv27
  %6 = load i8, i8* %scevgep55, align 1
  %idxprom.i.i = sext i8 %6 to i64
  %isascii.i.i224 = icmp sgt i8 %6, -1
  br i1 %isascii.i.i224, label %cond.true.i.i, label %cond.false.i.i, !prof !181
cond.true.i.i:
  %arrayidx.i.i = getelementptr inbounds %struct._RuneLocale, %struct._RuneLocale* @_DefaultRuneLocale, i64 0, i32 5, i64 %idxprom.i.i
  %7 = load i32, i32* %arrayidx.i.i, align 4
  %and.i.i = and i32 %7, 32768
  br label %isupper.exit
cond.false.i.i:
  %8 = trunc i64 %idxprom.i.i to i8
  %conv8 = sext i8 %8 to i32
  %call3.i.i = tail call i32 @__maskrune(i32 %conv8, i64 32768) #3
  br label %isupper.exit
isupper.exit:
  %tobool1.sink.i.in.i = phi i32 [ %and.i.i, %cond.true.i.i ], [ %call3.i.i, %cond.false.i.i ]
  %tobool1.sink.i.i = icmp eq i32 %tobool1.sink.i.in.i, 0
  br i1 %tobool1.sink.i.i, label %lor.rhs, label %while.body, !prof !989
lor.rhs:
  %sunkaddr = ptrtoint i8* %5 to i64
  %sunkaddr58 = add i64 %sunkaddr, %lsr.iv27
  %sunkaddr59 = inttoptr i64 %sunkaddr58 to i8*
  %9 = load i8, i8* %sunkaddr59, align 1
  %idxprom.i.i214 = sext i8 %9 to i64
  %isascii.i.i213225 = icmp sgt i8 %9, -1
  br i1 %isascii.i.i213225, label %cond.true.i.i217, label %cond.false.i.i219, !prof !181
cond.true.i.i217:
  %arrayidx.i.i215 = getelementptr inbounds %struct._RuneLocale, %struct._RuneLocale* @_DefaultRuneLocale, i64 0, i32 5, i64 %idxprom.i.i214
  %10 = load i32, i32* %arrayidx.i.i215, align 4
  %and.i.i216 = and i32 %10, 32768
  br label %isupper.exit223
cond.false.i.i219:
  %11 = trunc i64 %idxprom.i.i214 to i8
  %conv9 = sext i8 %11 to i32
  %call3.i.i218 = tail call i32 @__maskrune(i32 %conv9, i64 32768) #3
  br label %isupper.exit223
isupper.exit223:
  %tobool1.sink.i.in.i220 = phi i32 [ %and.i.i216, %cond.true.i.i217 ], [ %call3.i.i218, %cond.false.i.i219 ]
  %tobool1.sink.i.i221 = icmp eq i32 %tobool1.sink.i.in.i220, 0
  br i1 %tobool1.sink.i.i221, label %while.end, label %while.body, !prof !990
while.body:
  %sunkaddr60 = ptrtoint i8* %4 to i64
  %sunkaddr61 = add i64 %sunkaddr60, %lsr.iv27
  %sunkaddr62 = inttoptr i64 %sunkaddr61 to i8*
  %12 = load i8, i8* %sunkaddr62, align 1
  %sunkaddr63 = ptrtoint i8* %5 to i64
  %sunkaddr64 = add i64 %sunkaddr63, %lsr.iv27
  %sunkaddr65 = inttoptr i64 %sunkaddr64 to i8*
  %13 = load i8, i8* %sunkaddr65, align 1
  %cmp14 = icmp eq i8 %12, %13
  br i1 %cmp14, label %if.end17, label %return, !prof !991
if.end17:
  %lsr.iv.next28 = add i64 %lsr.iv27, 1
  br label %while.cond
while.end:
  %14 = or i8 %3, %2
  %15 = icmp eq i8 %14, 0
  br i1 %15, label %if.then23, label %if.else88, !prof !992
if.then23:
  %sunkaddr66 = ptrtoint %struct.Connector_struct* %a to i64
  %sunkaddr67 = add i64 %sunkaddr66, 16
  %sunkaddr68 = inttoptr i64 %sunkaddr67 to i8**
  %16 = load i8*, i8** %sunkaddr68, align 8
  %17 = load i8, i8* %16, align 1
  %cmp26 = icmp eq i8 %17, 83
  %sunkaddr69 = ptrtoint i8* %4 to i64
  %sunkaddr70 = add i64 %sunkaddr69, %lsr.iv27
  %sunkaddr71 = inttoptr i64 %sunkaddr70 to i8*
  %18 = load i8, i8* %sunkaddr71, align 1
  br i1 %cmp26, label %land.lhs.true28, label %while.cond59.preheader, !prof !993
land.lhs.true28:
  switch i8 %18, label %land.rhs.preheader [
    i8 112, label %land.lhs.true35
    i8 0, label %return
  ], !prof !994
land.lhs.true35:
  %sunkaddr72 = ptrtoint i8* %5 to i64
  %sunkaddr73 = add i64 %sunkaddr72, %lsr.iv27
  %sunkaddr74 = inttoptr i64 %sunkaddr73 to i8*
  %19 = load i8, i8* %sunkaddr74, align 1
  switch i8 %19, label %land.rhs.preheader [
    i8 112, label %land.lhs.true43
  ], !prof !995
land.lhs.true43:
  %20 = ptrtoint i8* %16 to i64
  %21 = sub i64 0, %20
  %scevgep52 = getelementptr i8, i8* %4, i64 %21
  %scevgep53 = getelementptr i8, i8* %scevgep52, i64 %lsr.iv27
  %scevgep54 = getelementptr i8, i8* %scevgep53, i64 -1
  %cmp45 = icmp eq i8* %scevgep54, null
  br i1 %cmp45, label %return, label %lor.lhs.false47, !prof !996
lor.lhs.false47:
  %22 = ptrtoint i8* %16 to i64
  %23 = sub i64 0, %22
  %scevgep47 = getelementptr i8, i8* %4, i64 %23
  %scevgep48 = getelementptr i8, i8* %scevgep47, i64 %lsr.iv27
  %scevgep49 = getelementptr i8, i8* %scevgep48, i64 -2
  %cmp50 = icmp eq i8* %scevgep49, null
  br i1 %cmp50, label %land.lhs.true52, label %while.cond59.preheader, !prof !997
land.lhs.true52:
  %sunkaddr75 = ptrtoint i8* %4 to i64
  %sunkaddr76 = add i64 %sunkaddr75, %lsr.iv27
  %sunkaddr77 = add i64 %sunkaddr76, -1
  %sunkaddr78 = inttoptr i64 %sunkaddr77 to i8*
  %24 = load i8, i8* %sunkaddr78, align 1
  %cmp55 = icmp eq i8 %24, 73
  %cmp61233 = icmp eq i8 %18, 0
  %or.cond265 = or i1 %cmp55, %cmp61233
  br i1 %or.cond265, label %return, label %land.rhs.preheader, !prof !998
while.cond59.preheader:
  %cmp61233.old = icmp eq i8 %18, 0
  br i1 %cmp61233.old, label %return, label %land.rhs.preheader, !prof !999
land.rhs.preheader:
  %scevgep33 = getelementptr i8, i8* %5, i64 %lsr.iv27
  %scevgep43 = getelementptr i8, i8* %4, i64 %lsr.iv27
  br label %land.rhs
land.rhs:
  %lsr.iv = phi i64 [ 0, %land.rhs.preheader ], [ %lsr.iv.next, %if.then83 ]
  %25 = phi i8 [ %27, %if.then83 ], [ %18, %land.rhs.preheader ]
  %scevgep34 = getelementptr i8, i8* %scevgep33, i64 %lsr.iv
  %26 = load i8, i8* %scevgep34, align 1
  %cmp64 = icmp eq i8 %26, 0
  br i1 %cmp64, label %return, label %while.body66, !prof !1000
while.body66:
  %cmp68 = icmp eq i8 %25, 42
  %cmp72 = icmp eq i8 %26, 42
  %or.cond = or i1 %cmp68, %cmp72
  br i1 %or.cond, label %if.then83, label %lor.lhs.false74, !prof !1001
lor.lhs.false74:
  %cmp77 = icmp ne i8 %25, %26
  %cmp81 = icmp eq i8 %25, 94
  %or.cond208 = or i1 %cmp77, %cmp81
  br i1 %or.cond208, label %return, label %if.then83, !prof !1002
if.then83:
  %scevgep44 = getelementptr i8, i8* %scevgep43, i64 %lsr.iv
  %scevgep45 = getelementptr i8, i8* %scevgep44, i64 1
  %27 = load i8, i8* %scevgep45, align 1
  %cmp61 = icmp eq i8 %27, 0
  %lsr.iv.next = add i64 %lsr.iv, 1
  br i1 %cmp61, label %return, label %land.rhs, !prof !999
if.else88:
  %cmp89 = icmp eq i8 %2, 1
  %cmp92 = icmp eq i8 %3, 2
  %or.cond159 = and i1 %cmp89, %cmp92
  br i1 %or.cond159, label %while.cond95.preheader, label %if.else123, !prof !1003
while.cond95.preheader:
  %sunkaddr79 = ptrtoint i8* %4 to i64
  %sunkaddr80 = add i64 %sunkaddr79, %lsr.iv27
  %sunkaddr81 = inttoptr i64 %sunkaddr80 to i8*
  %28 = load i8, i8* %sunkaddr81, align 1
  %cmp97238 = icmp eq i8 %28, 0
  br i1 %cmp97238, label %return, label %land.rhs99.preheader, !prof !1004
land.rhs99.preheader:
  %scevgep31 = getelementptr i8, i8* %5, i64 %lsr.iv27
  %scevgep40 = getelementptr i8, i8* %4, i64 %lsr.iv27
  br label %land.rhs99
land.rhs99:
  %lsr.iv17 = phi i64 [ 0, %land.rhs99.preheader ], [ %lsr.iv.next18, %if.then117 ]
  %29 = phi i8 [ %31, %if.then117 ], [ %28, %land.rhs99.preheader ]
  %scevgep32 = getelementptr i8, i8* %scevgep31, i64 %lsr.iv17
  %30 = load i8, i8* %scevgep32, align 1
  %cmp101 = icmp eq i8 %30, 0
  br i1 %cmp101, label %return, label %while.body104, !prof !1005
while.body104:
  %cmp107 = icmp eq i8 %29, %30
  %cmp111 = icmp eq i8 %29, 42
  %or.cond209 = or i1 %cmp107, %cmp111
  %cmp115 = icmp eq i8 %30, 94
  %or.cond210 = or i1 %or.cond209, %cmp115
  br i1 %or.cond210, label %if.then117, label %return, !prof !1006
if.then117:
  %scevgep41 = getelementptr i8, i8* %scevgep40, i64 %lsr.iv17
  %scevgep42 = getelementptr i8, i8* %scevgep41, i64 1
  %31 = load i8, i8* %scevgep42, align 1
  %cmp97 = icmp eq i8 %31, 0
  %lsr.iv.next18 = add i64 %lsr.iv17, 1
  br i1 %cmp97, label %return, label %land.rhs99, !prof !1004
if.else123:
  %cmp124 = icmp eq i8 %3, 1
  %cmp127 = icmp eq i8 %2, 2
  %or.cond160 = and i1 %cmp124, %cmp127
  br i1 %or.cond160, label %while.cond130.preheader, label %return, !prof !1007
while.cond130.preheader:
  %sunkaddr82 = ptrtoint i8* %4 to i64
  %sunkaddr83 = add i64 %sunkaddr82, %lsr.iv27
  %sunkaddr84 = inttoptr i64 %sunkaddr83 to i8*
  %32 = load i8, i8* %sunkaddr84, align 1
  %cmp132244 = icmp eq i8 %32, 0
  br i1 %cmp132244, label %return, label %land.rhs134.preheader, !prof !1008
land.rhs134.preheader:
  %scevgep29 = getelementptr i8, i8* %5, i64 %lsr.iv27
  %scevgep37 = getelementptr i8, i8* %4, i64 %lsr.iv27
  br label %land.rhs134
land.rhs134:
  %lsr.iv22 = phi i64 [ 0, %land.rhs134.preheader ], [ %lsr.iv.next23, %if.then152 ]
  %33 = phi i8 [ %35, %if.then152 ], [ %32, %land.rhs134.preheader ]
  %scevgep30 = getelementptr i8, i8* %scevgep29, i64 %lsr.iv22
  %34 = load i8, i8* %scevgep30, align 1
  %cmp136 = icmp eq i8 %34, 0
  br i1 %cmp136, label %return, label %while.body139, !prof !1009
while.body139:
  %cmp142 = icmp eq i8 %33, %34
  %cmp146 = icmp eq i8 %34, 42
  %or.cond211 = or i1 %cmp142, %cmp146
  %cmp150 = icmp eq i8 %33, 94
  %or.cond212 = or i1 %or.cond211, %cmp150
  br i1 %or.cond212, label %if.then152, label %return, !prof !1010
if.then152:
  %scevgep38 = getelementptr i8, i8* %scevgep37, i64 %lsr.iv22
  %scevgep39 = getelementptr i8, i8* %scevgep38, i64 1
  %35 = load i8, i8* %scevgep39, align 1
  %cmp132 = icmp eq i8 %35, 0
  %lsr.iv.next23 = add i64 %lsr.iv22, 1
  br i1 %cmp132, label %return, label %land.rhs134, !prof !1008
return:
  %retval.0 = phi i32 [ 0, %entry ], [ 1, %land.lhs.true52 ], [ 1, %land.lhs.true43 ], [ 0, %if.else123 ], [ 1, %while.cond59.preheader ], [ 1, %while.cond95.preheader ], [ 1, %while.cond130.preheader ], [ 1, %land.lhs.true28 ], [ 1, %if.then83 ], [ 0, %lor.lhs.false74 ], [ 1, %land.rhs ], [ 1, %if.then117 ], [ 0, %while.body104 ], [ 1, %land.rhs99 ], [ 1, %if.then152 ], [ 0, %while.body139 ], [ 1, %land.rhs134 ], [ 0, %while.body ]
  ret i32 %retval.0
}
!181 = !{!"branch_weights", i32 662038, i32 1}
!988 = !{!"branch_weights", i32 12091450, i32 1916}
!989 = !{!"branch_weights", i32 7564670, i32 4526781}
!990 = !{!"branch_weights", i32 7484958, i32 13283499}
!991 = !{!"branch_weights", i32 8677007, i32 4606493}
!992 = !{!"branch_weights", i32 -1172426948, i32 145094705}
!993 = !{!"branch_weights", i32 1468914, i32 5683688}
!994 = !{!"branch_weights", i32 114025221, i32 -1217548794, i32 -1199521551, i32 87712616}
!995 = !{!"branch_weights", i32 1853716452, i32 -444717951, i32 932776759}
!996 = !{!"branch_weights", i32 1004870, i32 20259}
!997 = !{!"branch_weights", i32 20071, i32 189}
!998 = !{!"branch_weights", i32 -1020255939, i32 572177766}
!999 = !{!"branch_weights", i32 2666513, i32 3466431}
!1000 = !{!"branch_weights", i32 5117635, i32 1859780}
!1001 = !{!"branch_weights", i32 354902465, i32 -1444604407}
!1002 = !{!"branch_weights", i32 -1762419279, i32 1592770684}
!1003 = !{!"branch_weights", i32 1435905930, i32 -1951930624}
!1004 = !{!"branch_weights", i32 1, i32 504888}
!1005 = !{!"branch_weights", i32 94662, i32 504888}
!1006 = !{!"branch_weights", i32 -1897793104, i32 160196332}
!1007 = !{!"branch_weights", i32 2074643678, i32 -29579071}
!1008 = !{!"branch_weights", i32 1, i32 226163}
!1009 = !{!"branch_weights", i32 58357, i32 226163}
!1010 = !{!"branch_weights", i32 -2072848646, i32 92907517}
