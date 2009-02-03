; RUN: llvm-as < %s | opt -globalsmodref-aa -dse -disable-output
target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8"
	%struct.ECacheType = type { i32, i32, i32 }
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.QTType = type { i8, i8, i16, i32, i32, i32 }
	%struct.TType = type { i8, i8, i8, i8, i16, i32, i32, i32 }
	%struct._RuneEntry = type { i32, i32, i32, i32* }
	%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i32, i8**)*, i32 (i32, i8*, i32, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32 }
	%struct._RuneRange = type { i32, %struct._RuneEntry* }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.move_s = type { i32, i32, i32, i32, i32, i32 }
	%struct.move_x = type { i32, i32, i32, i32 }
	%struct.node_t = type { i8, i8, i8, i8, i32, i32, %struct.node_t**, %struct.node_t*, %struct.move_s }
	%struct.see_data = type { i32, i32 }
@rook_o.2925 = internal global [4 x i32] [ i32 12, i32 -12, i32 1, i32 -1 ]		; <[4 x i32]*> [#uses=0]
@bishop_o.2926 = internal global [4 x i32] [ i32 11, i32 -11, i32 13, i32 -13 ]		; <[4 x i32]*> [#uses=0]
@knight_o.2927 = internal global [8 x i32] [ i32 10, i32 -10, i32 14, i32 -14, i32 23, i32 -23, i32 25, i32 -25 ]		; <[8 x i32]*> [#uses=0]
@board = internal global [144 x i32] zeroinitializer		; <[144 x i32]*> [#uses=0]
@holding = internal global [2 x [16 x i32]] zeroinitializer		; <[2 x [16 x i32]]*> [#uses=0]
@hold_hash = internal global i32 0		; <i32*> [#uses=0]
@white_hand_eval = internal global i32 0		; <i32*> [#uses=0]
@black_hand_eval = internal global i32 0		; <i32*> [#uses=0]
@num_holding = internal global [2 x i32] zeroinitializer		; <[2 x i32]*> [#uses=0]
@zobrist = internal global [14 x [144 x i32]] zeroinitializer		; <[14 x [144 x i32]]*> [#uses=0]
@Variant = internal global i32 0		; <i32*> [#uses=7]
@userealholdings.b = internal global i1 false		; <i1*> [#uses=1]
@realholdings = internal global [255 x i8] zeroinitializer		; <[255 x i8]*> [#uses=0]
@comp_color = internal global i32 0		; <i32*> [#uses=0]
@C.97.3177 = internal global [13 x i32] [ i32 0, i32 2, i32 1, i32 4, i32 3, i32 0, i32 0, i32 8, i32 7, i32 10, i32 9, i32 12, i32 11 ]		; <[13 x i32]*> [#uses=0]
@str = internal global [30 x i8] c"%s:%u: failed assertion `%s'\0A\00"		; <[30 x i8]*> [#uses=0]
@str.upgrd.1 = internal global [81 x i8] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/crazy.c\00"		; <[81 x i8]*> [#uses=0]
@str.upgrd.2 = internal global [32 x i8] c"piece > frame && piece < npiece\00"		; <[32 x i8]*> [#uses=0]
@C.101.3190 = internal global [13 x i32] [ i32 0, i32 2, i32 1, i32 2, i32 1, i32 0, i32 0, i32 2, i32 1, i32 2, i32 1, i32 2, i32 1 ]		; <[13 x i32]*> [#uses=0]
@hand_value = internal global [13 x i32] [ i32 0, i32 100, i32 -100, i32 210, i32 -210, i32 0, i32 0, i32 250, i32 -250, i32 450, i32 -450, i32 230, i32 -230 ]		; <[13 x i32]*> [#uses=0]
@material = internal global [14 x i32] zeroinitializer		; <[14 x i32]*> [#uses=0]
@Material = internal global i32 0		; <i32*> [#uses=0]
@str.upgrd.3 = internal global [23 x i8] c"holding[who][what] > 0\00"		; <[23 x i8]*> [#uses=0]
@str.upgrd.4 = internal global [24 x i8] c"holding[who][what] < 20\00"		; <[24 x i8]*> [#uses=0]
@fifty = internal global i32 0		; <i32*> [#uses=0]
@move_number = internal global i32 0		; <i32*> [#uses=1]
@ply = internal global i32 0		; <i32*> [#uses=2]
@hash_history = internal global [600 x i32] zeroinitializer		; <[600 x i32]*> [#uses=1]
@hash = internal global i32 0		; <i32*> [#uses=1]
@ECacheSize.b = internal global i1 false		; <i1*> [#uses=1]
@ECache = internal global %struct.ECacheType* null		; <%struct.ECacheType**> [#uses=1]
@ECacheProbes = internal global i32 0		; <i32*> [#uses=1]
@ECacheHits = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.5 = internal global [34 x i8] c"Out of memory allocating ECache.\0A\00"		; <[34 x i8]*> [#uses=0]
@rankoffsets.2930 = internal global [8 x i32] [ i32 110, i32 98, i32 86, i32 74, i32 62, i32 50, i32 38, i32 26 ]		; <[8 x i32]*> [#uses=0]
@white_castled = internal global i32 0		; <i32*> [#uses=0]
@black_castled = internal global i32 0		; <i32*> [#uses=0]
@book_ply = internal global i32 0		; <i32*> [#uses=0]
@bking_loc = internal global i32 0		; <i32*> [#uses=1]
@wking_loc = internal global i32 0		; <i32*> [#uses=1]
@white_to_move = internal global i32 0		; <i32*> [#uses=3]
@moved = internal global [144 x i32] zeroinitializer		; <[144 x i32]*> [#uses=0]
@ep_square = internal global i32 0		; <i32*> [#uses=0]
@_DefaultRuneLocale = external global %struct._RuneLocale		; <%struct._RuneLocale*> [#uses=0]
@str.upgrd.6 = internal global [3 x i8] c"bm\00"		; <[3 x i8]*> [#uses=0]
@str1 = internal global [3 x i8] c"am\00"		; <[3 x i8]*> [#uses=0]
@str1.upgrd.7 = internal global [34 x i8] c"No best-move or avoid-move found!\00"		; <[34 x i8]*> [#uses=0]
@str.upgrd.8 = internal global [25 x i8] c"\0AName of EPD testsuite: \00"		; <[25 x i8]*> [#uses=0]
@__sF = external global [0 x %struct.FILE]		; <[0 x %struct.FILE]*> [#uses=0]
@str.upgrd.9 = internal global [21 x i8] c"\0ATime per move (s): \00"		; <[21 x i8]*> [#uses=0]
@str.upgrd.10 = internal global [2 x i8] c"\0A\00"		; <[2 x i8]*> [#uses=0]
@str2 = internal global [2 x i8] c"r\00"		; <[2 x i8]*> [#uses=0]
@root_to_move = internal global i32 0		; <i32*> [#uses=1]
@forcedwin.b = internal global i1 false		; <i1*> [#uses=2]
@fixed_time = internal global i32 0		; <i32*> [#uses=1]
@nodes = internal global i32 0		; <i32*> [#uses=1]
@qnodes = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.11 = internal global [29 x i8] c"\0ANodes: %i (%0.2f%% qnodes)\0A\00"		; <[29 x i8]*> [#uses=0]
@str.upgrd.12 = internal global [54 x i8] c"ECacheProbes : %u   ECacheHits : %u   HitRate : %f%%\0A\00"		; <[54 x i8]*> [#uses=0]
@TTStores = internal global i32 0		; <i32*> [#uses=1]
@TTProbes = internal global i32 0		; <i32*> [#uses=1]
@TTHits = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.13 = internal global [60 x i8] c"TTStores : %u TTProbes : %u   TTHits : %u   HitRate : %f%%\0A\00"		; <[60 x i8]*> [#uses=0]
@NTries = internal global i32 0		; <i32*> [#uses=1]
@NCuts = internal global i32 0		; <i32*> [#uses=1]
@TExt = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.14 = internal global [51 x i8] c"NTries : %u  NCuts : %u  CutRate : %f%%  TExt: %u\0A\00"		; <[51 x i8]*> [#uses=0]
@ext_check = internal global i32 0		; <i32*> [#uses=1]
@razor_drop = internal global i32 0		; <i32*> [#uses=1]
@razor_material = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.15 = internal global [61 x i8] c"Check extensions: %u  Razor drops : %u  Razor Material : %u\0A\00"		; <[61 x i8]*> [#uses=0]
@FHF = internal global i32 0		; <i32*> [#uses=1]
@FH = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.16 = internal global [22 x i8] c"Move ordering : %f%%\0A\00"		; <[22 x i8]*> [#uses=0]
@maxposdiff = internal global i32 0		; <i32*> [#uses=1]
@str.upgrd.17 = internal global [47 x i8] c"Material score: %d  Eval : %d  MaxPosDiff: %d\0A\00"		; <[47 x i8]*> [#uses=0]
@str.upgrd.18 = internal global [17 x i8] c"Solution found.\0A\00"		; <[17 x i8]*> [#uses=0]
@str3 = internal global [21 x i8] c"Solution not found.\0A\00"		; <[21 x i8]*> [#uses=0]
@str.upgrd.19 = internal global [15 x i8] c"Solved: %d/%d\0A\00"		; <[15 x i8]*> [#uses=0]
@str.upgrd.20 = internal global [9 x i8] c"EPD: %s\0A\00"		; <[9 x i8]*> [#uses=0]
@str4 = internal global [21 x i8] c"Searching to %d ply\0A\00"		; <[21 x i8]*> [#uses=0]
@maxdepth = internal global i32 0		; <i32*> [#uses=0]
@std_material = internal global [14 x i32] [ i32 0, i32 100, i32 -100, i32 310, i32 -310, i32 4000, i32 -4000, i32 500, i32 -500, i32 900, i32 -900, i32 325, i32 -325, i32 0 ]		; <[14 x i32]*> [#uses=0]
@zh_material = internal global [14 x i32] [ i32 0, i32 100, i32 -100, i32 210, i32 -210, i32 4000, i32 -4000, i32 250, i32 -250, i32 450, i32 -450, i32 230, i32 -230, i32 0 ]		; <[14 x i32]*> [#uses=0]
@suicide_material = internal global [14 x i32] [ i32 0, i32 15, i32 -15, i32 150, i32 -150, i32 500, i32 -500, i32 150, i32 -150, i32 50, i32 -50, i32 0, i32 0, i32 0 ]		; <[14 x i32]*> [#uses=0]
@losers_material = internal global [14 x i32] [ i32 0, i32 80, i32 -80, i32 320, i32 -320, i32 1000, i32 -1000, i32 350, i32 -350, i32 400, i32 -400, i32 270, i32 -270, i32 0 ]		; <[14 x i32]*> [#uses=0]
@Xfile = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@Xrank = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 0, i32 0, i32 0, i32 0, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 0, i32 0, i32 0, i32 0, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@Xdiagl = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 9, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 10, i32 9, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 0, i32 0, i32 0, i32 0, i32 11, i32 10, i32 9, i32 1, i32 2, i32 3, i32 4, i32 5, i32 0, i32 0, i32 0, i32 0, i32 12, i32 11, i32 10, i32 9, i32 1, i32 2, i32 3, i32 4, i32 0, i32 0, i32 0, i32 0, i32 13, i32 12, i32 11, i32 10, i32 9, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 1, i32 2, i32 0, i32 0, i32 0, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@Xdiagr = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 1, i32 0, i32 0, i32 0, i32 0, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 1, i32 2, i32 0, i32 0, i32 0, i32 0, i32 13, i32 12, i32 11, i32 10, i32 9, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 12, i32 11, i32 10, i32 9, i32 1, i32 2, i32 3, i32 4, i32 0, i32 0, i32 0, i32 0, i32 11, i32 10, i32 9, i32 1, i32 2, i32 3, i32 4, i32 5, i32 0, i32 0, i32 0, i32 0, i32 10, i32 9, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 0, i32 0, i32 0, i32 0, i32 9, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@sqcolor = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@pcsqbishop = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 -5, i32 -10, i32 -5, i32 -5, i32 -10, i32 -5, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 10, i32 5, i32 10, i32 10, i32 5, i32 10, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 5, i32 6, i32 15, i32 15, i32 6, i32 5, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 3, i32 15, i32 10, i32 10, i32 15, i32 3, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 3, i32 15, i32 10, i32 10, i32 15, i32 3, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 5, i32 6, i32 15, i32 15, i32 6, i32 5, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 10, i32 5, i32 10, i32 10, i32 5, i32 10, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 -5, i32 -10, i32 -5, i32 -5, i32 -10, i32 -5, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@black_knight = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 15, i32 25, i32 25, i32 25, i32 25, i32 15, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 15, i32 25, i32 35, i32 35, i32 35, i32 15, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 10, i32 25, i32 20, i32 25, i32 25, i32 10, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 20, i32 20, i32 20, i32 20, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 15, i32 15, i32 15, i32 15, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 3, i32 3, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -35, i32 -10, i32 -10, i32 -10, i32 -10, i32 -35, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@white_knight = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -35, i32 -10, i32 -10, i32 -10, i32 -10, i32 -35, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 3, i32 3, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 15, i32 15, i32 15, i32 15, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 20, i32 20, i32 20, i32 20, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 10, i32 25, i32 20, i32 25, i32 25, i32 10, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 15, i32 25, i32 35, i32 35, i32 35, i32 15, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 15, i32 25, i32 25, i32 25, i32 25, i32 15, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@white_pawn = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 25, i32 25, i32 35, i32 5, i32 5, i32 50, i32 45, i32 30, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 14, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 20, i32 20, i32 10, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 12, i32 18, i32 18, i32 27, i32 27, i32 18, i32 18, i32 18, i32 0, i32 0, i32 0, i32 0, i32 25, i32 30, i32 30, i32 35, i32 35, i32 35, i32 30, i32 25, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@black_pawn = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 30, i32 30, i32 30, i32 35, i32 35, i32 35, i32 30, i32 25, i32 0, i32 0, i32 0, i32 0, i32 12, i32 18, i32 18, i32 27, i32 27, i32 18, i32 18, i32 18, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 20, i32 20, i32 10, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 14, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 25, i32 25, i32 35, i32 5, i32 5, i32 50, i32 45, i32 30, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@white_king = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -100, i32 7, i32 4, i32 0, i32 10, i32 4, i32 7, i32 -100, i32 0, i32 0, i32 0, i32 0, i32 -250, i32 -200, i32 -150, i32 -100, i32 -100, i32 -150, i32 -200, i32 -250, i32 0, i32 0, i32 0, i32 0, i32 -350, i32 -300, i32 -300, i32 -250, i32 -250, i32 -300, i32 -300, i32 -350, i32 0, i32 0, i32 0, i32 0, i32 -400, i32 -400, i32 -400, i32 -350, i32 -350, i32 -400, i32 -400, i32 -400, i32 0, i32 0, i32 0, i32 0, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 0, i32 0, i32 0, i32 0, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 0, i32 0, i32 0, i32 0, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 0, i32 0, i32 0, i32 0, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@black_king = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 0, i32 0, i32 0, i32 0, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 0, i32 0, i32 0, i32 0, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 -500, i32 0, i32 0, i32 0, i32 0, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 -450, i32 0, i32 0, i32 0, i32 0, i32 -400, i32 -400, i32 -400, i32 -350, i32 -350, i32 -400, i32 -400, i32 -400, i32 0, i32 0, i32 0, i32 0, i32 -350, i32 -300, i32 -300, i32 -250, i32 -250, i32 -300, i32 -300, i32 -350, i32 0, i32 0, i32 0, i32 0, i32 -250, i32 -200, i32 -150, i32 -100, i32 -100, i32 -150, i32 -200, i32 -250, i32 0, i32 0, i32 0, i32 0, i32 -100, i32 7, i32 4, i32 0, i32 10, i32 4, i32 7, i32 -100, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@black_queen = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 5, i32 5, i32 10, i32 10, i32 5, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 3, i32 3, i32 3, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 0, i32 0, i32 0, i32 0, i32 -60, i32 -40, i32 -40, i32 -60, i32 -60, i32 -40, i32 -40, i32 -60, i32 0, i32 0, i32 0, i32 0, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 -15, i32 -15, i32 -10, i32 -10, i32 -15, i32 -15, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 10, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@white_queen = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 10, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 -15, i32 -15, i32 -10, i32 -10, i32 -15, i32 -15, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 -40, i32 0, i32 0, i32 0, i32 0, i32 -60, i32 -40, i32 -40, i32 -60, i32 -60, i32 -40, i32 -40, i32 -60, i32 0, i32 0, i32 0, i32 0, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 -30, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 3, i32 3, i32 3, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 5, i32 5, i32 10, i32 10, i32 5, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@black_rook = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 15, i32 20, i32 25, i32 25, i32 20, i32 15, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 15, i32 20, i32 20, i32 15, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -20, i32 -20, i32 -30, i32 -30, i32 -20, i32 -20, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 -15, i32 -15, i32 -10, i32 -10, i32 -15, i32 -15, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@white_rook = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 -15, i32 -15, i32 -10, i32 -10, i32 -15, i32 -15, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -20, i32 -20, i32 -30, i32 -30, i32 -20, i32 -20, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 15, i32 20, i32 20, i32 15, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 15, i32 20, i32 25, i32 25, i32 20, i32 15, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@upscale = internal global [64 x i32] [ i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117 ]		; <[64 x i32]*> [#uses=0]
@pre_p_tropism = internal global [9 x i32] [ i32 9999, i32 40, i32 20, i32 10, i32 3, i32 1, i32 1, i32 0, i32 9999 ]		; <[9 x i32]*> [#uses=0]
@pre_r_tropism = internal global [9 x i32] [ i32 9999, i32 50, i32 40, i32 15, i32 5, i32 1, i32 1, i32 0, i32 9999 ]		; <[9 x i32]*> [#uses=0]
@pre_n_tropism = internal global [9 x i32] [ i32 9999, i32 50, i32 70, i32 35, i32 10, i32 2, i32 1, i32 0, i32 9999 ]		; <[9 x i32]*> [#uses=0]
@pre_q_tropism = internal global [9 x i32] [ i32 9999, i32 100, i32 60, i32 20, i32 5, i32 2, i32 0, i32 0, i32 9999 ]		; <[9 x i32]*> [#uses=0]
@pre_b_tropism = internal global [9 x i32] [ i32 9999, i32 50, i32 25, i32 15, i32 5, i32 2, i32 2, i32 2, i32 9999 ]		; <[9 x i32]*> [#uses=0]
@rookdistance = internal global [144 x [144 x i32]] zeroinitializer		; <[144 x [144 x i32]]*> [#uses=0]
@distance = internal global [144 x [144 x i32]] zeroinitializer		; <[144 x [144 x i32]]*> [#uses=0]
@p_tropism = internal global [144 x [144 x i8]] zeroinitializer		; <[144 x [144 x i8]]*> [#uses=0]
@b_tropism = internal global [144 x [144 x i8]] zeroinitializer		; <[144 x [144 x i8]]*> [#uses=0]
@n_tropism = internal global [144 x [144 x i8]] zeroinitializer		; <[144 x [144 x i8]]*> [#uses=0]
@r_tropism = internal global [144 x [144 x i8]] zeroinitializer		; <[144 x [144 x i8]]*> [#uses=0]
@q_tropism = internal global [144 x [144 x i8]] zeroinitializer		; <[144 x [144 x i8]]*> [#uses=0]
@cfg_devscale.b = internal global i1 false		; <i1*> [#uses=0]
@pieces = internal global [62 x i32] zeroinitializer		; <[62 x i32]*> [#uses=0]
@piece_count = internal global i32 0		; <i32*> [#uses=1]
@cfg_smarteval.b = internal global i1 false		; <i1*> [#uses=0]
@lcentral = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -15, i32 -15, i32 -15, i32 -15, i32 -15, i32 -15, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 0, i32 3, i32 5, i32 5, i32 3, i32 0, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 0, i32 15, i32 15, i32 15, i32 15, i32 0, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 0, i32 15, i32 30, i32 30, i32 15, i32 0, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 0, i32 15, i32 30, i32 30, i32 15, i32 0, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 0, i32 15, i32 15, i32 15, i32 15, i32 0, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -15, i32 0, i32 3, i32 5, i32 5, i32 3, i32 0, i32 -15, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -15, i32 -15, i32 -15, i32 -15, i32 -15, i32 -15, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@str3.upgrd.21 = internal global [81 x i8] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/leval.c\00"		; <[81 x i8]*> [#uses=0]
@str5 = internal global [21 x i8] c"(i > 0) && (i < 145)\00"		; <[21 x i8]*> [#uses=0]
@kingcap.b = internal global i1 false		; <i1*> [#uses=0]
@numb_moves = internal global i32 0		; <i32*> [#uses=2]
@genfor = internal global %struct.move_s* null		; <%struct.move_s**> [#uses=0]
@captures = internal global i32 0		; <i32*> [#uses=1]
@fcaptures.b = internal global i1 false		; <i1*> [#uses=0]
@gfrom = internal global i32 0		; <i32*> [#uses=0]
@Giveaway.b = internal global i1 false		; <i1*> [#uses=0]
@path_x = internal global [300 x %struct.move_x] zeroinitializer		; <[300 x %struct.move_x]*> [#uses=0]
@str7 = internal global [81 x i8] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/moves.c\00"		; <[81 x i8]*> [#uses=0]
@str8 = internal global [15 x i8] c"find_slot < 63\00"		; <[15 x i8]*> [#uses=0]
@is_promoted = internal global [62 x i32] zeroinitializer		; <[62 x i32]*> [#uses=0]
@squares = internal global [144 x i32] zeroinitializer		; <[144 x i32]*> [#uses=0]
@str.upgrd.22 = internal global [38 x i8] c"promoted > frame && promoted < npiece\00"		; <[38 x i8]*> [#uses=0]
@str1.upgrd.23 = internal global [38 x i8] c"promoted < npiece && promoted > frame\00"		; <[38 x i8]*> [#uses=0]
@evalRoutines = internal global [7 x i32 (i32, i32)*] [ i32 (i32, i32)* @ErrorIt, i32 (i32, i32)* @Pawn, i32 (i32, i32)* @Knight, i32 (i32, i32)* @King, i32 (i32, i32)* @Rook, i32 (i32, i32)* @Queen, i32 (i32, i32)* @Bishop ]		; <[7 x i32 (i32, i32)*]*> [#uses=0]
@sbishop = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 8, i32 5, i32 5, i32 5, i32 5, i32 8, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 3, i32 3, i32 5, i32 5, i32 3, i32 3, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 2, i32 5, i32 4, i32 4, i32 5, i32 2, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 2, i32 5, i32 4, i32 4, i32 5, i32 2, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 3, i32 3, i32 5, i32 5, i32 3, i32 3, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 8, i32 5, i32 5, i32 5, i32 5, i32 8, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 -2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@sknight = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 3, i32 3, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 5, i32 5, i32 5, i32 5, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 5, i32 10, i32 10, i32 5, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 5, i32 10, i32 10, i32 5, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 5, i32 5, i32 5, i32 5, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 3, i32 3, i32 0, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@swhite_pawn = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 10, i32 10, i32 3, i32 2, i32 1, i32 0, i32 0, i32 0, i32 0, i32 2, i32 4, i32 6, i32 12, i32 12, i32 6, i32 4, i32 2, i32 0, i32 0, i32 0, i32 0, i32 3, i32 6, i32 9, i32 14, i32 14, i32 9, i32 6, i32 3, i32 0, i32 0, i32 0, i32 0, i32 10, i32 12, i32 14, i32 16, i32 16, i32 14, i32 12, i32 10, i32 0, i32 0, i32 0, i32 0, i32 20, i32 22, i32 24, i32 26, i32 26, i32 24, i32 22, i32 20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@sblack_pawn = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 20, i32 22, i32 24, i32 26, i32 26, i32 24, i32 22, i32 20, i32 0, i32 0, i32 0, i32 0, i32 10, i32 12, i32 14, i32 16, i32 16, i32 14, i32 12, i32 10, i32 0, i32 0, i32 0, i32 0, i32 3, i32 6, i32 9, i32 14, i32 14, i32 9, i32 6, i32 3, i32 0, i32 0, i32 0, i32 0, i32 2, i32 4, i32 6, i32 12, i32 12, i32 6, i32 4, i32 2, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2, i32 3, i32 10, i32 10, i32 3, i32 2, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@swhite_king = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 14, i32 0, i32 0, i32 0, i32 9, i32 14, i32 2, i32 0, i32 0, i32 0, i32 0, i32 -3, i32 -5, i32 -6, i32 -6, i32 -6, i32 -6, i32 -5, i32 -3, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 -5, i32 -8, i32 -8, i32 -8, i32 -8, i32 -5, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -8, i32 -8, i32 -13, i32 -13, i32 -13, i32 -13, i32 -8, i32 -8, i32 0, i32 0, i32 0, i32 0, i32 -13, i32 -13, i32 -21, i32 -21, i32 -21, i32 -21, i32 -13, i32 -13, i32 0, i32 0, i32 0, i32 0, i32 -21, i32 -21, i32 -34, i32 -34, i32 -34, i32 -34, i32 -21, i32 -21, i32 0, i32 0, i32 0, i32 0, i32 -34, i32 -34, i32 -55, i32 -55, i32 -55, i32 -55, i32 -34, i32 -34, i32 0, i32 0, i32 0, i32 0, i32 -55, i32 -55, i32 -89, i32 -89, i32 -89, i32 -89, i32 -55, i32 -55, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@sblack_king = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -55, i32 -55, i32 -89, i32 -89, i32 -89, i32 -89, i32 -55, i32 -55, i32 0, i32 0, i32 0, i32 0, i32 -34, i32 -34, i32 -55, i32 -55, i32 -55, i32 -55, i32 -34, i32 -34, i32 0, i32 0, i32 0, i32 0, i32 -21, i32 -21, i32 -34, i32 -34, i32 -34, i32 -34, i32 -21, i32 -21, i32 0, i32 0, i32 0, i32 0, i32 -13, i32 -13, i32 -21, i32 -21, i32 -21, i32 -21, i32 -13, i32 -13, i32 0, i32 0, i32 0, i32 0, i32 -8, i32 -8, i32 -13, i32 -13, i32 -13, i32 -13, i32 -8, i32 -8, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 -5, i32 -8, i32 -8, i32 -8, i32 -8, i32 -5, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -3, i32 -5, i32 -6, i32 -6, i32 -6, i32 -6, i32 -5, i32 -3, i32 0, i32 0, i32 0, i32 0, i32 2, i32 14, i32 0, i32 0, i32 0, i32 9, i32 14, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@send_king = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 -3, i32 -1, i32 0, i32 0, i32 -1, i32 -3, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 -3, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 -3, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 10, i32 25, i32 25, i32 25, i32 25, i32 10, i32 -1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 25, i32 50, i32 50, i32 25, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 25, i32 50, i32 50, i32 25, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 10, i32 25, i32 25, i32 25, i32 25, i32 10, i32 -1, i32 0, i32 0, i32 0, i32 0, i32 -3, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 -3, i32 0, i32 0, i32 0, i32 0, i32 -5, i32 -3, i32 -1, i32 0, i32 0, i32 -1, i32 -3, i32 -5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@srev_rank = internal global [9 x i32] [ i32 0, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1 ]		; <[9 x i32]*> [#uses=0]
@std_p_tropism = internal global [8 x i32] [ i32 9999, i32 15, i32 10, i32 7, i32 2, i32 0, i32 0, i32 0 ]		; <[8 x i32]*> [#uses=0]
@std_own_p_tropism = internal global [8 x i32] [ i32 9999, i32 30, i32 10, i32 2, i32 0, i32 0, i32 0, i32 0 ]		; <[8 x i32]*> [#uses=0]
@std_r_tropism = internal global [16 x i32] [ i32 9999, i32 0, i32 15, i32 5, i32 2, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[16 x i32]*> [#uses=0]
@std_n_tropism = internal global [8 x i32] [ i32 9999, i32 14, i32 9, i32 6, i32 1, i32 0, i32 0, i32 0 ]		; <[8 x i32]*> [#uses=0]
@std_q_tropism = internal global [8 x i32] [ i32 9999, i32 200, i32 50, i32 15, i32 3, i32 2, i32 1, i32 0 ]		; <[8 x i32]*> [#uses=0]
@std_b_tropism = internal global [8 x i32] [ i32 9999, i32 12, i32 7, i32 5, i32 0, i32 0, i32 0, i32 0 ]		; <[8 x i32]*> [#uses=0]
@phase = internal global i32 0		; <i32*> [#uses=1]
@dir.3001 = internal global [4 x i32] [ i32 -13, i32 -11, i32 11, i32 13 ]		; <[4 x i32]*> [#uses=0]
@dir.3021 = internal global [4 x i32] [ i32 -1, i32 1, i32 12, i32 -12 ]		; <[4 x i32]*> [#uses=0]
@king_locs = internal global [2 x i32] zeroinitializer		; <[2 x i32]*> [#uses=0]
@square_d1.3081 = internal global [2 x i32] [ i32 29, i32 113 ]		; <[2 x i32]*> [#uses=0]
@wmat = internal global i32 0		; <i32*> [#uses=0]
@bmat = internal global i32 0		; <i32*> [#uses=0]
@str.upgrd.24 = internal global [35 x i8] c"Illegal piece detected sq=%i c=%i\0A\00"		; <[35 x i8]*> [#uses=0]
@str10 = internal global [81 x i8] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/neval.c\00"		; <[81 x i8]*> [#uses=0]
@std_hand_value = internal global [13 x i32] [ i32 0, i32 100, i32 -100, i32 210, i32 -210, i32 0, i32 0, i32 250, i32 -250, i32 450, i32 -450, i32 230, i32 -230 ]		; <[13 x i32]*> [#uses=0]
@xb_mode = internal global i32 0		; <i32*> [#uses=0]
@str.upgrd.25 = internal global [69 x i8] c"tellics ptell Hello! I am Sjeng and hope you enjoy playing with me.\0A\00"		; <[69 x i8]*> [#uses=0]
@str.upgrd.26 = internal global [76 x i8] c"tellics ptell For help on some commands that I understand, ptell me 'help'\0A\00"		; <[76 x i8]*> [#uses=0]
@str12 = internal global [3 x i8] c"%s\00"		; <[3 x i8]*> [#uses=0]
@my_partner = internal global [256 x i8] zeroinitializer		; <[256 x i8]*> [#uses=0]
@str13 = internal global [25 x i8] c"tellics set f5 bughouse\0A\00"		; <[25 x i8]*> [#uses=0]
@str.upgrd.27 = internal global [16 x i8] c"tellics unseek\0A\00"		; <[16 x i8]*> [#uses=0]
@str.upgrd.28 = internal global [20 x i8] c"tellics set f5 1=1\0A\00"		; <[20 x i8]*> [#uses=0]
@str.upgrd.29 = internal global [80 x i8] c"is...uh...what did you say?\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"		; <[80 x i8]*> [#uses=0]
@str.upgrd.30 = internal global [5 x i8] c"help\00"		; <[5 x i8]*> [#uses=0]
@str.upgrd.31 = internal global [147 x i8] c"tellics ptell Commands that I understand are : sit, go, fast, slow, abort, flag, +/++/+++/-/--/---{p,n,b,r,q,d,h,trades}, x, dead, formula, help.\0A\00"		; <[147 x i8]*> [#uses=0]
@str.upgrd.32 = internal global [6 x i8] c"sorry\00"		; <[6 x i8]*> [#uses=0]
@str.upgrd.33 = internal global [59 x i8] c"tellics ptell Sorry, but I'm not playing a bughouse game.\0A\00"		; <[59 x i8]*> [#uses=0]
@str.upgrd.34 = internal global [4 x i8] c"sit\00"		; <[4 x i8]*> [#uses=0]
@str.upgrd.35 = internal global [56 x i8] c"tellics ptell Ok, I sit next move. Tell me when to go.\0A\00"		; <[56 x i8]*> [#uses=0]
@must_sit.b = internal global i1 false		; <i1*> [#uses=0]
@str114 = internal global [3 x i8] c"go\00"		; <[3 x i8]*> [#uses=0]
@str2.upgrd.36 = internal global [5 x i8] c"move\00"		; <[5 x i8]*> [#uses=0]
@str.upgrd.37 = internal global [31 x i8] c"tellics ptell Ok, I'm moving.\0A\00"		; <[31 x i8]*> [#uses=0]
@str3.upgrd.38 = internal global [5 x i8] c"fast\00"		; <[5 x i8]*> [#uses=0]
@str4.upgrd.39 = internal global [5 x i8] c"time\00"		; <[5 x i8]*> [#uses=0]
@str15 = internal global [35 x i8] c"tellics ptell Ok, I'm going FAST!\0A\00"		; <[35 x i8]*> [#uses=0]
@go_fast.b = internal global i1 false		; <i1*> [#uses=0]
@str5.upgrd.40 = internal global [5 x i8] c"slow\00"		; <[5 x i8]*> [#uses=0]
@str16 = internal global [36 x i8] c"tellics ptell Ok, moving normally.\0A\00"		; <[36 x i8]*> [#uses=0]
@str6 = internal global [6 x i8] c"abort\00"		; <[6 x i8]*> [#uses=0]
@str7.upgrd.41 = internal global [35 x i8] c"tellics ptell Requesting abort...\0A\00"		; <[35 x i8]*> [#uses=0]
@str17 = internal global [15 x i8] c"tellics abort\0A\00"		; <[15 x i8]*> [#uses=0]
@str8.upgrd.42 = internal global [5 x i8] c"flag\00"		; <[5 x i8]*> [#uses=0]
@str.upgrd.43 = internal global [27 x i8] c"tellics ptell Flagging...\0A\00"		; <[27 x i8]*> [#uses=0]
@str.upgrd.44 = internal global [14 x i8] c"tellics flag\0A\00"		; <[14 x i8]*> [#uses=0]
@str18 = internal global [2 x i8] c"+\00"		; <[2 x i8]*> [#uses=0]
@str9 = internal global [6 x i8] c"trade\00"		; <[6 x i8]*> [#uses=0]
@str10.upgrd.45 = internal global [35 x i8] c"tellics ptell Ok, trading is GOOD\0A\00"		; <[35 x i8]*> [#uses=0]
@str11 = internal global [4 x i8] c"+++\00"		; <[4 x i8]*> [#uses=0]
@str12.upgrd.46 = internal global [6 x i8] c"mates\00"		; <[6 x i8]*> [#uses=0]
@str13.upgrd.47 = internal global [3 x i8] c"++\00"		; <[3 x i8]*> [#uses=0]
@str.upgrd.48 = internal global [49 x i8] c"is VERY good (ptell me 'x' to play normal again)\00"		; <[49 x i8]*> [#uses=0]
@str.upgrd.49 = internal global [44 x i8] c"is good (ptell me 'x' to play normal again)\00"		; <[44 x i8]*> [#uses=0]
@str19 = internal global [29 x i8] c"tellics ptell Ok, Knight %s\0A\00"		; <[29 x i8]*> [#uses=0]
@str14 = internal global [29 x i8] c"tellics ptell Ok, Bishop %s\0A\00"		; <[29 x i8]*> [#uses=0]
@str15.upgrd.50 = internal global [27 x i8] c"tellics ptell Ok, Rook %s\0A\00"		; <[27 x i8]*> [#uses=0]
@str.upgrd.51 = internal global [28 x i8] c"tellics ptell Ok, Queen %s\0A\00"		; <[28 x i8]*> [#uses=0]
@str16.upgrd.52 = internal global [27 x i8] c"tellics ptell Ok, Pawn %s\0A\00"		; <[27 x i8]*> [#uses=0]
@str17.upgrd.53 = internal global [31 x i8] c"tellics ptell Ok, Diagonal %s\0A\00"		; <[31 x i8]*> [#uses=0]
@str18.upgrd.54 = internal global [28 x i8] c"tellics ptell Ok, Heavy %s\0A\00"		; <[28 x i8]*> [#uses=0]
@str20 = internal global [34 x i8] c"tellics ptell Ok, trading is BAD\0A\00"		; <[34 x i8]*> [#uses=0]
@str20.upgrd.55 = internal global [4 x i8] c"---\00"		; <[4 x i8]*> [#uses=0]
@str.upgrd.56 = internal global [53 x i8] c"mates you (ptell me 'x' when it no longer mates you)\00"		; <[53 x i8]*> [#uses=0]
@str21 = internal global [3 x i8] c"--\00"		; <[3 x i8]*> [#uses=0]
@str.upgrd.57 = internal global [52 x i8] c"is VERY bad (ptell me 'x' when it is no longer bad)\00"		; <[52 x i8]*> [#uses=0]
@str21.upgrd.58 = internal global [47 x i8] c"is bad (ptell me 'x' when it is no longer bad)\00"		; <[47 x i8]*> [#uses=0]
@str23 = internal global [16 x i8] c"mate me anymore\00"		; <[16 x i8]*> [#uses=0]
@str24 = internal global [6 x i8] c"never\00"		; <[6 x i8]*> [#uses=0]
@str25 = internal global [5 x i8] c"mind\00"		; <[5 x i8]*> [#uses=0]
@str22 = internal global [9 x i8] c"ptell me\00"		; <[9 x i8]*> [#uses=0]
@str.upgrd.59 = internal global [55 x i8] c"tellics ptell Ok, reverting to STANDARD piece values!\0A\00"		; <[55 x i8]*> [#uses=0]
@partnerdead.b = internal global i1 false		; <i1*> [#uses=0]
@piecedead.b = internal global i1 false		; <i1*> [#uses=0]
@str.upgrd.60 = internal global [26 x i8] c"i'll have to sit...(dead)\00"		; <[26 x i8]*> [#uses=0]
@str27 = internal global [5 x i8] c"dead\00"		; <[5 x i8]*> [#uses=0]
@str28 = internal global [27 x i8] c"i'll have to sit...(piece)\00"		; <[27 x i8]*> [#uses=0]
@str29 = internal global [3 x i8] c"ok\00"		; <[3 x i8]*> [#uses=0]
@str30 = internal global [3 x i8] c"hi\00"		; <[3 x i8]*> [#uses=0]
@str31 = internal global [6 x i8] c"hello\00"		; <[6 x i8]*> [#uses=0]
@str32 = internal global [26 x i8] c"tellics ptell Greetings.\0A\00"		; <[26 x i8]*> [#uses=0]
@str.upgrd.61 = internal global [8 x i8] c"formula\00"		; <[8 x i8]*> [#uses=0]
@str.upgrd.62 = internal global [87 x i8] c"tellics ptell Setting formula, if you are still interrupted, complain to my operator.\0A\00"		; <[87 x i8]*> [#uses=0]
@str33 = internal global [59 x i8] c"tellics ptell Sorry, but I don't understand that command.\0A\00"		; <[59 x i8]*> [#uses=0]
@pawnmated.3298 = internal global i32 0		; <i32*> [#uses=0]
@knightmated.3299 = internal global i32 0		; <i32*> [#uses=0]
@bishopmated.3300 = internal global i32 0		; <i32*> [#uses=0]
@rookmated.3301 = internal global i32 0		; <i32*> [#uses=0]
@queenmated.3302 = internal global i32 0		; <i32*> [#uses=0]
@str.upgrd.63 = internal global [41 x i8] c"tellics ptell p doesn't mate me anymore\0A\00"		; <[41 x i8]*> [#uses=0]
@str34 = internal global [41 x i8] c"tellics ptell n doesn't mate me anymore\0A\00"		; <[41 x i8]*> [#uses=0]
@str35 = internal global [41 x i8] c"tellics ptell b doesn't mate me anymore\0A\00"		; <[41 x i8]*> [#uses=0]
@str36 = internal global [41 x i8] c"tellics ptell r doesn't mate me anymore\0A\00"		; <[41 x i8]*> [#uses=0]
@str37 = internal global [41 x i8] c"tellics ptell q doesn't mate me anymore\0A\00"		; <[41 x i8]*> [#uses=0]
@str38 = internal global [20 x i8] c"tellics ptell ---p\0A\00"		; <[20 x i8]*> [#uses=0]
@str39 = internal global [20 x i8] c"tellics ptell ---n\0A\00"		; <[20 x i8]*> [#uses=0]
@str40 = internal global [20 x i8] c"tellics ptell ---b\0A\00"		; <[20 x i8]*> [#uses=0]
@str41 = internal global [20 x i8] c"tellics ptell ---r\0A\00"		; <[20 x i8]*> [#uses=0]
@str42 = internal global [20 x i8] c"tellics ptell ---q\0A\00"		; <[20 x i8]*> [#uses=0]
@str23.upgrd.64 = internal global [17 x i8] c"tellics ptell x\0A\00"		; <[17 x i8]*> [#uses=0]
@str.upgrd.65 = internal global [18 x i8] c"tellics ptell go\0A\00"		; <[18 x i8]*> [#uses=0]
@bufftop = internal global i32 0		; <i32*> [#uses=2]
@membuff = internal global i8* null		; <i8**> [#uses=3]
@maxply = internal global i32 0		; <i32*> [#uses=1]
@forwards = internal global i32 0		; <i32*> [#uses=1]
@nodecount = internal global i32 0		; <i32*> [#uses=1]
@frees = internal global i32 0		; <i32*> [#uses=0]
@PBSize.b = internal global i1 false		; <i1*> [#uses=1]
@alllosers.b = internal global i1 false		; <i1*> [#uses=1]
@rootlosers = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=1]
@pn_move = internal global %struct.move_s zeroinitializer		; <%struct.move_s*> [#uses=7]
@iters = internal global i32 0		; <i32*> [#uses=1]
@kibitzed.b = internal global i1 false		; <i1*> [#uses=0]
@str24.upgrd.66 = internal global [28 x i8] c"tellics kibitz Forced win!\0A\00"		; <[28 x i8]*> [#uses=0]
@str25.upgrd.67 = internal global [34 x i8] c"tellics kibitz Forced win! (alt)\0A\00"		; <[34 x i8]*> [#uses=0]
@pn_time = internal global i32 0		; <i32*> [#uses=1]
@post = internal global i32 0		; <i32*> [#uses=0]
@str.upgrd.68 = internal global [94 x i8] c"tellics whisper proof %d, disproof %d, %d losers, highest depth %d, primary %d, secondary %d\0A\00"		; <[94 x i8]*> [#uses=0]
@str26 = internal global [30 x i8] c"tellics whisper Forced reply\0A\00"		; <[30 x i8]*> [#uses=0]
@str27.upgrd.69 = internal global [60 x i8] c"P: %d D: %d N: %d S: %d Mem: %2.2fM Iters: %d MaxDepth: %d\0A\00"		; <[60 x i8]*> [#uses=0]
@str.upgrd.70 = internal global [90 x i8] c"tellics whisper proof %d, disproof %d, %d nodes, %d forwards, %d iters, highest depth %d\0A\00"		; <[90 x i8]*> [#uses=0]
@str.upgrd.71 = internal global [11 x i8] c"Time : %f\0A\00"		; <[11 x i8]*> [#uses=0]
@str28.upgrd.72 = internal global [23 x i8] c"This position is WON.\0A\00"		; <[23 x i8]*> [#uses=0]
@str29.upgrd.73 = internal global [5 x i8] c"PV: \00"		; <[5 x i8]*> [#uses=0]
@str30.upgrd.74 = internal global [4 x i8] c"%s \00"		; <[4 x i8]*> [#uses=0]
@str31.upgrd.75 = internal global [2 x i8] c" \00"		; <[2 x i8]*> [#uses=0]
@str32.upgrd.76 = internal global [41 x i8] c"\0Atellics kibitz Forced win in %d moves.\0A\00"		; <[41 x i8]*> [#uses=0]
@str33.upgrd.77 = internal global [20 x i8] c"\0A1-0 {White mates}\0A\00"		; <[20 x i8]*> [#uses=0]
@result = internal global i32 0		; <i32*> [#uses=4]
@str1.upgrd.78 = internal global [20 x i8] c"\0A0-1 {Black mates}\0A\00"		; <[20 x i8]*> [#uses=0]
@str35.upgrd.79 = internal global [24 x i8] c"This position is LOST.\0A\00"		; <[24 x i8]*> [#uses=0]
@str36.upgrd.80 = internal global [27 x i8] c"This position is UNKNOWN.\0A\00"		; <[27 x i8]*> [#uses=0]
@str37.upgrd.81 = internal global [47 x i8] c"P: %d D: %d N: %d S: %d Mem: %2.2fM Iters: %d\0A\00"		; <[47 x i8]*> [#uses=0]
@s_threat.b = internal global i1 false		; <i1*> [#uses=0]
@TTSize.b = internal global i1 false		; <i1*> [#uses=3]
@cfg_razordrop.b = internal global i1 false		; <i1*> [#uses=0]
@cfg_futprune.b = internal global i1 false		; <i1*> [#uses=0]
@cfg_onerep.b = internal global i1 false		; <i1*> [#uses=0]
@setcode = internal global [30 x i8] zeroinitializer		; <[30 x i8]*> [#uses=0]
@str38.upgrd.82 = internal global [3 x i8] c"%u\00"		; <[3 x i8]*> [#uses=0]
@searching_pv.b = internal global i1 false		; <i1*> [#uses=0]
@pv = internal global [300 x [300 x %struct.move_s]] zeroinitializer		; <[300 x [300 x %struct.move_s]]*> [#uses=0]
@i_depth = internal global i32 0		; <i32*> [#uses=0]
@history_h = internal global [144 x [144 x i32]] zeroinitializer		; <[144 x [144 x i32]]*> [#uses=0]
@killer1 = internal global [300 x %struct.move_s] zeroinitializer		; <[300 x %struct.move_s]*> [#uses=0]
@killer2 = internal global [300 x %struct.move_s] zeroinitializer		; <[300 x %struct.move_s]*> [#uses=0]
@killer3 = internal global [300 x %struct.move_s] zeroinitializer		; <[300 x %struct.move_s]*> [#uses=0]
@rootnodecount = internal global [512 x i32] zeroinitializer		; <[512 x i32]*> [#uses=0]
@raw_nodes = internal global i32 0		; <i32*> [#uses=0]
@pv_length = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@time_exit.b = internal global i1 false		; <i1*> [#uses=0]
@time_for_move = internal global i32 0		; <i32*> [#uses=3]
@failed = internal global i32 0		; <i32*> [#uses=0]
@extendedtime.b = internal global i1 false		; <i1*> [#uses=1]
@time_left = internal global i32 0		; <i32*> [#uses=0]
@str39.upgrd.83 = internal global [38 x i8] c"Extended from %d to %d, time left %d\0A\00"		; <[38 x i8]*> [#uses=0]
@checks = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@singular = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@recaps = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@ext_onerep = internal global i32 0		; <i32*> [#uses=1]
@FULL = internal global i32 0		; <i32*> [#uses=1]
@PVS = internal global i32 0		; <i32*> [#uses=1]
@PVSF = internal global i32 0		; <i32*> [#uses=1]
@killer_scores = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@killer_scores2 = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@killer_scores3 = internal global [300 x i32] zeroinitializer		; <[300 x i32]*> [#uses=0]
@time_failure.b = internal global i1 false		; <i1*> [#uses=0]
@cur_score = internal global i32 0		; <i32*> [#uses=0]
@legals = internal global i32 0		; <i32*> [#uses=3]
@movetotal = internal global i32 0		; <i32*> [#uses=0]
@searching_move = internal global [20 x i8] zeroinitializer		; <[20 x i8]*> [#uses=0]
@is_pondering.b = internal global i1 false		; <i1*> [#uses=6]
@true_i_depth = internal global i8 0		; <i8*> [#uses=1]
@is_analyzing.b = internal global i1 false		; <i1*> [#uses=0]
@inc = internal global i32 0		; <i32*> [#uses=1]
@time_cushion = internal global i32 0		; <i32*> [#uses=2]
@str40.upgrd.84 = internal global [16 x i8] c"Opening phase.\0A\00"		; <[16 x i8]*> [#uses=1]
@str.upgrd.85 = internal global [19 x i8] c"Middlegame phase.\0A\00"		; <[19 x i8]*> [#uses=1]
@str1.upgrd.86 = internal global [16 x i8] c"Endgame phase.\0A\00"		; <[16 x i8]*> [#uses=1]
@str43 = internal global [20 x i8] c"Time for move : %d\0A\00"		; <[20 x i8]*> [#uses=1]
@postpv = internal global [256 x i8] zeroinitializer		; <[256 x i8]*> [#uses=0]
@str44 = internal global [49 x i8] c"tellics whisper %d restart(s), ended up with %s\0A\00"		; <[49 x i8]*> [#uses=0]
@moves_to_tc = internal global i32 0		; <i32*> [#uses=0]
@str45 = internal global [27 x i8] c"tellics kibitz Mate in %d\0A\00"		; <[27 x i8]*> [#uses=0]
@str46 = internal global [52 x i8] c"tellics ptell Mate in %d, give him no more pieces.\0A\00"		; <[52 x i8]*> [#uses=0]
@tradefreely.b = internal global i1 false		; <i1*> [#uses=0]
@str.upgrd.87 = internal global [37 x i8] c"tellics ptell You can trade freely.\0A\00"		; <[37 x i8]*> [#uses=0]
@str47 = internal global [25 x i8] c"tellics ptell ---trades\0A\00"		; <[25 x i8]*> [#uses=0]
@str2.upgrd.88 = internal global [49 x i8] c"tellics kibitz Both players dead...resigning...\0A\00"		; <[49 x i8]*> [#uses=0]
@str3.upgrd.89 = internal global [16 x i8] c"tellics resign\0A\00"		; <[16 x i8]*> [#uses=0]
@str48 = internal global [81 x i8] c"tellics ptell I am forcedly mated (dead). Tell me 'go' to start moving into it.\0A\00"		; <[81 x i8]*> [#uses=0]
@str.upgrd.90 = internal global [62 x i8] c"tellics ptell I'll have to sit...(lose piece that mates you)\0A\00"		; <[62 x i8]*> [#uses=0]
@see_num_attackers = internal global [2 x i32] zeroinitializer		; <[2 x i32]*> [#uses=0]
@see_attackers = internal global [2 x [16 x %struct.see_data]] zeroinitializer		; <[2 x [16 x %struct.see_data]]*> [#uses=0]
@scentral = internal global [144 x i32] [ i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 3, i32 5, i32 5, i32 3, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 2, i32 15, i32 15, i32 15, i32 15, i32 2, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 7, i32 15, i32 25, i32 25, i32 15, i32 7, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 7, i32 15, i32 25, i32 25, i32 15, i32 7, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 2, i32 15, i32 15, i32 15, i32 15, i32 2, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -10, i32 0, i32 3, i32 5, i32 5, i32 3, i32 0, i32 -10, i32 0, i32 0, i32 0, i32 0, i32 -20, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -10, i32 -20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ]		; <[144 x i32]*> [#uses=0]
@str51 = internal global [81 x i8] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/seval.c\00"		; <[81 x i8]*> [#uses=0]
@divider = internal global [50 x i8] c"-------------------------------------------------\00"		; <[50 x i8]*> [#uses=0]
@min_per_game = internal global i32 0		; <i32*> [#uses=0]
@opp_rating = internal global i32 0		; <i32*> [#uses=0]
@my_rating = internal global i32 0		; <i32*> [#uses=0]
@str53 = internal global [15 x i8] c"SPEC Workload\0A\00"		; <[15 x i8]*> [#uses=0]
@opening_history = internal global [256 x i8] zeroinitializer		; <[256 x i8]*> [#uses=0]
@str60 = internal global [81 x i8] c"Material score: %d   Eval : %d  MaxPosDiff: %d  White hand: %d  Black hand : %d\0A\00"		; <[81 x i8]*> [#uses=0]
@str61 = internal global [26 x i8] c"Hash : %X  HoldHash : %X\0A\00"		; <[26 x i8]*> [#uses=0]
@str62 = internal global [9 x i8] c"move %s\0A\00"		; <[9 x i8]*> [#uses=0]
@str63 = internal global [5 x i8] c"\0A%s\0A\00"		; <[5 x i8]*> [#uses=0]
@str64 = internal global [19 x i8] c"0-1 {Black Mates}\0A\00"		; <[19 x i8]*> [#uses=0]
@str1.upgrd.91 = internal global [19 x i8] c"1-0 {White Mates}\0A\00"		; <[19 x i8]*> [#uses=0]
@str65 = internal global [27 x i8] c"1/2-1/2 {Fifty move rule}\0A\00"		; <[27 x i8]*> [#uses=0]
@str2.upgrd.92 = internal global [29 x i8] c"1/2-1/2 {3 fold repetition}\0A\00"		; <[29 x i8]*> [#uses=0]
@str66 = internal global [16 x i8] c"1/2-1/2 {Draw}\0A\00"		; <[16 x i8]*> [#uses=0]
@str68 = internal global [8 x i8] c"Sjeng: \00"		; <[8 x i8]*> [#uses=0]
@str69 = internal global [18 x i8] c"Illegal move: %s\0A\00"		; <[18 x i8]*> [#uses=0]
@str3.upgrd.93 = internal global [9 x i8] c"setboard\00"		; <[9 x i8]*> [#uses=0]
@str470 = internal global [5 x i8] c"quit\00"		; <[5 x i8]*> [#uses=0]
@str571 = internal global [5 x i8] c"exit\00"		; <[5 x i8]*> [#uses=0]
@str6.upgrd.94 = internal global [8 x i8] c"diagram\00"		; <[8 x i8]*> [#uses=0]
@str7.upgrd.95 = internal global [2 x i8] c"d\00"		; <[2 x i8]*> [#uses=0]
@str72 = internal global [6 x i8] c"perft\00"		; <[6 x i8]*> [#uses=0]
@str73 = internal global [3 x i8] c"%d\00"		; <[3 x i8]*> [#uses=0]
@str74 = internal global [28 x i8] c"Raw nodes for depth %d: %i\0A\00"		; <[28 x i8]*> [#uses=0]
@str.upgrd.96 = internal global [13 x i8] c"Time : %.2f\0A\00"		; <[13 x i8]*> [#uses=0]
@str75 = internal global [4 x i8] c"new\00"		; <[4 x i8]*> [#uses=0]
@str.upgrd.97 = internal global [40 x i8] c"tellics set 1 Sjeng SPEC 1.0 (SPEC/%s)\0A\00"		; <[40 x i8]*> [#uses=0]
@str.upgrd.98 = internal global [7 x i8] c"xboard\00"		; <[7 x i8]*> [#uses=0]
@str8.upgrd.99 = internal global [6 x i8] c"nodes\00"		; <[6 x i8]*> [#uses=0]
@str77 = internal global [38 x i8] c"Number of nodes: %i (%0.2f%% qnodes)\0A\00"		; <[38 x i8]*> [#uses=0]
@str9.upgrd.100 = internal global [5 x i8] c"post\00"		; <[5 x i8]*> [#uses=0]
@str10.upgrd.101 = internal global [7 x i8] c"nopost\00"		; <[7 x i8]*> [#uses=0]
@str11.upgrd.102 = internal global [7 x i8] c"random\00"		; <[7 x i8]*> [#uses=0]
@str12.upgrd.103 = internal global [5 x i8] c"hard\00"		; <[5 x i8]*> [#uses=0]
@str13.upgrd.104 = internal global [5 x i8] c"easy\00"		; <[5 x i8]*> [#uses=0]
@str14.upgrd.105 = internal global [2 x i8] c"?\00"		; <[2 x i8]*> [#uses=0]
@str15.upgrd.106 = internal global [6 x i8] c"white\00"		; <[6 x i8]*> [#uses=0]
@str16.upgrd.107 = internal global [6 x i8] c"black\00"		; <[6 x i8]*> [#uses=0]
@str17.upgrd.108 = internal global [6 x i8] c"force\00"		; <[6 x i8]*> [#uses=0]
@str18.upgrd.109 = internal global [5 x i8] c"eval\00"		; <[5 x i8]*> [#uses=0]
@str.upgrd.110 = internal global [10 x i8] c"Eval: %d\0A\00"		; <[10 x i8]*> [#uses=0]
@str2178 = internal global [3 x i8] c"%i\00"		; <[3 x i8]*> [#uses=0]
@str22.upgrd.111 = internal global [5 x i8] c"otim\00"		; <[5 x i8]*> [#uses=0]
@opp_time = internal global i32 0		; <i32*> [#uses=0]
@str23.upgrd.112 = internal global [6 x i8] c"level\00"		; <[6 x i8]*> [#uses=0]
@str.upgrd.113 = internal global [12 x i8] c"%i %i:%i %i\00"		; <[12 x i8]*> [#uses=0]
@sec_per_game = internal global i32 0		; <i32*> [#uses=0]
@str24.upgrd.114 = internal global [9 x i8] c"%i %i %i\00"		; <[9 x i8]*> [#uses=0]
@str25.upgrd.115 = internal global [7 x i8] c"rating\00"		; <[7 x i8]*> [#uses=0]
@str26.upgrd.116 = internal global [6 x i8] c"%i %i\00"		; <[6 x i8]*> [#uses=0]
@str27.upgrd.117 = internal global [8 x i8] c"holding\00"		; <[8 x i8]*> [#uses=0]
@str28.upgrd.118 = internal global [8 x i8] c"variant\00"		; <[8 x i8]*> [#uses=0]
@str29.upgrd.119 = internal global [7 x i8] c"normal\00"		; <[7 x i8]*> [#uses=0]
@str79 = internal global [11 x i8] c"crazyhouse\00"		; <[11 x i8]*> [#uses=0]
@str30.upgrd.120 = internal global [9 x i8] c"bughouse\00"		; <[9 x i8]*> [#uses=0]
@str31.upgrd.121 = internal global [8 x i8] c"suicide\00"		; <[8 x i8]*> [#uses=0]
@str32.upgrd.122 = internal global [9 x i8] c"giveaway\00"		; <[9 x i8]*> [#uses=0]
@str33.upgrd.123 = internal global [7 x i8] c"losers\00"		; <[7 x i8]*> [#uses=0]
@str34.upgrd.124 = internal global [8 x i8] c"analyze\00"		; <[8 x i8]*> [#uses=0]
@str35.upgrd.125 = internal global [5 x i8] c"undo\00"		; <[5 x i8]*> [#uses=0]
@str36.upgrd.126 = internal global [18 x i8] c"Move number : %d\0A\00"		; <[18 x i8]*> [#uses=0]
@str37.upgrd.127 = internal global [7 x i8] c"remove\00"		; <[7 x i8]*> [#uses=0]
@str38.upgrd.128 = internal global [5 x i8] c"edit\00"		; <[5 x i8]*> [#uses=0]
@str41.upgrd.129 = internal global [2 x i8] c"#\00"		; <[2 x i8]*> [#uses=0]
@str42.upgrd.130 = internal global [8 x i8] c"partner\00"		; <[8 x i8]*> [#uses=0]
@str43.upgrd.131 = internal global [9 x i8] c"$partner\00"		; <[9 x i8]*> [#uses=0]
@str44.upgrd.132 = internal global [6 x i8] c"ptell\00"		; <[6 x i8]*> [#uses=0]
@str45.upgrd.133 = internal global [5 x i8] c"test\00"		; <[5 x i8]*> [#uses=0]
@str46.upgrd.134 = internal global [3 x i8] c"st\00"		; <[3 x i8]*> [#uses=0]
@str47.upgrd.135 = internal global [7 x i8] c"result\00"		; <[7 x i8]*> [#uses=0]
@str48.upgrd.136 = internal global [6 x i8] c"prove\00"		; <[6 x i8]*> [#uses=0]
@str49 = internal global [26 x i8] c"\0AMax time to search (s): \00"		; <[26 x i8]*> [#uses=0]
@str50 = internal global [5 x i8] c"ping\00"		; <[5 x i8]*> [#uses=0]
@str51.upgrd.137 = internal global [9 x i8] c"pong %d\0A\00"		; <[9 x i8]*> [#uses=0]
@str52 = internal global [6 x i8] c"fritz\00"		; <[6 x i8]*> [#uses=0]
@str53.upgrd.138 = internal global [6 x i8] c"reset\00"		; <[6 x i8]*> [#uses=0]
@str54 = internal global [3 x i8] c"sd\00"		; <[3 x i8]*> [#uses=0]
@str55 = internal global [26 x i8] c"New max depth set to: %d\0A\00"		; <[26 x i8]*> [#uses=0]
@str56 = internal global [5 x i8] c"auto\00"		; <[5 x i8]*> [#uses=0]
@str57 = internal global [9 x i8] c"protover\00"		; <[9 x i8]*> [#uses=0]
@str.upgrd.139 = internal global [63 x i8] c"feature ping=0 setboard=1 playother=0 san=0 usermove=0 time=1\0A\00"		; <[63 x i8]*> [#uses=0]
@str80 = internal global [53 x i8] c"feature draw=0 sigint=0 sigterm=0 reuse=1 analyze=0\0A\00"		; <[53 x i8]*> [#uses=0]
@str.upgrd.140 = internal global [33 x i8] c"feature myname=\22Sjeng SPEC 1.0\22\0A\00"		; <[33 x i8]*> [#uses=0]
@str.upgrd.141 = internal global [71 x i8] c"feature variants=\22normal,bughouse,crazyhouse,suicide,giveaway,losers\22\0A\00"		; <[71 x i8]*> [#uses=0]
@str.upgrd.142 = internal global [46 x i8] c"feature colors=1 ics=0 name=0 pause=0 done=1\0A\00"		; <[46 x i8]*> [#uses=0]
@str58 = internal global [9 x i8] c"accepted\00"		; <[9 x i8]*> [#uses=0]
@str59 = internal global [9 x i8] c"rejected\00"		; <[9 x i8]*> [#uses=0]
@str.upgrd.143 = internal global [65 x i8] c"Interface does not support a required feature...expect trouble.\0A\00"		; <[65 x i8]*> [#uses=0]
@str61.upgrd.144 = internal global [6 x i8] c"\0A%s\0A\0A\00"		; <[6 x i8]*> [#uses=0]
@str81 = internal global [41 x i8] c"diagram/d:       toggle diagram display\0A\00"		; <[41 x i8]*> [#uses=0]
@str82 = internal global [34 x i8] c"exit/quit:       terminate Sjeng\0A\00"		; <[34 x i8]*> [#uses=0]
@str62.upgrd.145 = internal global [51 x i8] c"go:              make Sjeng play the side to move\0A\00"		; <[51 x i8]*> [#uses=0]
@str83 = internal global [35 x i8] c"new:             start a new game\0A\00"		; <[35 x i8]*> [#uses=0]
@str84 = internal global [55 x i8] c"level <x>:       the xboard style command to set time\0A\00"		; <[55 x i8]*> [#uses=0]
@str85 = internal global [49 x i8] c"  <x> should be in the form: <a> <b> <c> where:\0A\00"		; <[49 x i8]*> [#uses=0]
@str63.upgrd.146 = internal global [49 x i8] c"  a -> moves to TC (0 if using an ICS style TC)\0A\00"		; <[49 x i8]*> [#uses=0]
@str86 = internal global [25 x i8] c"  b -> minutes per game\0A\00"		; <[25 x i8]*> [#uses=0]
@str64.upgrd.147 = internal global [29 x i8] c"  c -> increment in seconds\0A\00"		; <[29 x i8]*> [#uses=0]
@str65.upgrd.148 = internal global [55 x i8] c"nodes:           outputs the number of nodes searched\0A\00"		; <[55 x i8]*> [#uses=0]
@str87 = internal global [47 x i8] c"perft <x>:       compute raw nodes to depth x\0A\00"		; <[47 x i8]*> [#uses=0]
@str.upgrd.149 = internal global [42 x i8] c"post:            toggles thinking output\0A\00"		; <[42 x i8]*> [#uses=0]
@str.upgrd.150 = internal global [45 x i8] c"xboard:          put Sjeng into xboard mode\0A\00"		; <[45 x i8]*> [#uses=0]
@str.upgrd.151 = internal global [39 x i8] c"test:            run an EPD testsuite\0A\00"		; <[39 x i8]*> [#uses=0]
@str88 = internal global [52 x i8] c"speed:           test movegen and evaluation speed\0A\00"		; <[52 x i8]*> [#uses=0]
@str89 = internal global [59 x i8] c"proof:           try to prove or disprove the current pos\0A\00"		; <[59 x i8]*> [#uses=0]
@str90 = internal global [44 x i8] c"sd <x>:          limit thinking to depth x\0A\00"		; <[44 x i8]*> [#uses=0]
@str66.upgrd.152 = internal global [51 x i8] c"st <x>:          limit thinking to x centiseconds\0A\00"		; <[51 x i8]*> [#uses=0]
@str67 = internal global [54 x i8] c"setboard <FEN>:  set board to a specified FEN string\0A\00"		; <[54 x i8]*> [#uses=0]
@str68.upgrd.153 = internal global [38 x i8] c"undo:            back up a half move\0A\00"		; <[38 x i8]*> [#uses=0]
@str69.upgrd.154 = internal global [38 x i8] c"remove:          back up a full move\0A\00"		; <[38 x i8]*> [#uses=0]
@str70 = internal global [42 x i8] c"force:           disable computer moving\0A\00"		; <[42 x i8]*> [#uses=0]
@str71 = internal global [44 x i8] c"auto:            computer plays both sides\0A\00"		; <[44 x i8]*> [#uses=0]
@DP_TTable = internal global %struct.TType* null		; <%struct.TType**> [#uses=1]
@AS_TTable = internal global %struct.TType* null		; <%struct.TType**> [#uses=1]
@QS_TTable = internal global %struct.QTType* null		; <%struct.QTType**> [#uses=1]
@str93 = internal global [38 x i8] c"Out of memory allocating hashtables.\0A\00"		; <[38 x i8]*> [#uses=0]
@type_to_char.3058 = internal global [14 x i32] [ i32 70, i32 80, i32 80, i32 78, i32 78, i32 75, i32 75, i32 82, i32 82, i32 81, i32 81, i32 66, i32 66, i32 69 ]		; <[14 x i32]*> [#uses=0]
@str94 = internal global [8 x i8] c"%c@%c%d\00"		; <[8 x i8]*> [#uses=0]
@str95 = internal global [5 x i8] c"%c%d\00"		; <[5 x i8]*> [#uses=0]
@str1.upgrd.155 = internal global [8 x i8] c"%c%d=%c\00"		; <[8 x i8]*> [#uses=0]
@str2.upgrd.156 = internal global [8 x i8] c"%cx%c%d\00"		; <[8 x i8]*> [#uses=0]
@str96 = internal global [11 x i8] c"%cx%c%d=%c\00"		; <[11 x i8]*> [#uses=0]
@str97 = internal global [4 x i8] c"O-O\00"		; <[4 x i8]*> [#uses=0]
@str98 = internal global [6 x i8] c"O-O-O\00"		; <[6 x i8]*> [#uses=0]
@str99 = internal global [9 x i8] c"%c%c%c%d\00"		; <[9 x i8]*> [#uses=0]
@str3100 = internal global [9 x i8] c"%c%d%c%d\00"		; <[9 x i8]*> [#uses=0]
@str101 = internal global [10 x i8] c"%c%cx%c%d\00"		; <[10 x i8]*> [#uses=0]
@str4.upgrd.157 = internal global [10 x i8] c"%c%dx%c%d\00"		; <[10 x i8]*> [#uses=0]
@str102 = internal global [7 x i8] c"%c%c%d\00"		; <[7 x i8]*> [#uses=0]
@str5103 = internal global [5 x i8] c"illg\00"		; <[5 x i8]*> [#uses=0]
@type_to_char.3190 = internal global [14 x i32] [ i32 70, i32 80, i32 112, i32 78, i32 110, i32 75, i32 107, i32 82, i32 114, i32 81, i32 113, i32 66, i32 98, i32 69 ]		; <[14 x i32]*> [#uses=0]
@str7.upgrd.158 = internal global [10 x i8] c"%c%d%c%dn\00"		; <[10 x i8]*> [#uses=0]
@str8.upgrd.159 = internal global [10 x i8] c"%c%d%c%dr\00"		; <[10 x i8]*> [#uses=0]
@str9.upgrd.160 = internal global [10 x i8] c"%c%d%c%db\00"		; <[10 x i8]*> [#uses=0]
@str10.upgrd.161 = internal global [10 x i8] c"%c%d%c%dk\00"		; <[10 x i8]*> [#uses=0]
@str11.upgrd.162 = internal global [10 x i8] c"%c%d%c%dq\00"		; <[10 x i8]*> [#uses=0]
@C.88.3251 = internal global [14 x i8*] [ i8* getelementptr ([3 x i8]* @str105, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str12106, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str13107, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str141, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str152, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str163, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str174, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str185, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str19108, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str206, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str21109, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str227, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str238, i32 0, i32 0), i8* getelementptr ([3 x i8]* @str249, i32 0, i32 0) ]		; <[14 x i8*]*> [#uses=0]
@str105 = internal global [3 x i8] c"!!\00"		; <[3 x i8]*> [#uses=1]
@str12106 = internal global [3 x i8] c" P\00"		; <[3 x i8]*> [#uses=1]
@str13107 = internal global [3 x i8] c"*P\00"		; <[3 x i8]*> [#uses=1]
@str141 = internal global [3 x i8] c" N\00"		; <[3 x i8]*> [#uses=1]
@str152 = internal global [3 x i8] c"*N\00"		; <[3 x i8]*> [#uses=1]
@str163 = internal global [3 x i8] c" K\00"		; <[3 x i8]*> [#uses=1]
@str174 = internal global [3 x i8] c"*K\00"		; <[3 x i8]*> [#uses=1]
@str185 = internal global [3 x i8] c" R\00"		; <[3 x i8]*> [#uses=1]
@str19108 = internal global [3 x i8] c"*R\00"		; <[3 x i8]*> [#uses=1]
@str206 = internal global [3 x i8] c" Q\00"		; <[3 x i8]*> [#uses=1]
@str21109 = internal global [3 x i8] c"*Q\00"		; <[3 x i8]*> [#uses=1]
@str227 = internal global [3 x i8] c" B\00"		; <[3 x i8]*> [#uses=1]
@str238 = internal global [3 x i8] c"*B\00"		; <[3 x i8]*> [#uses=1]
@str249 = internal global [3 x i8] c"  \00"		; <[3 x i8]*> [#uses=1]
@str110 = internal global [42 x i8] c"+----+----+----+----+----+----+----+----+\00"		; <[42 x i8]*> [#uses=0]
@str25.upgrd.163 = internal global [6 x i8] c"  %s\0A\00"		; <[6 x i8]*> [#uses=0]
@str26.upgrd.164 = internal global [5 x i8] c"%d |\00"		; <[5 x i8]*> [#uses=0]
@str27.upgrd.165 = internal global [6 x i8] c" %s |\00"		; <[6 x i8]*> [#uses=0]
@str28.upgrd.166 = internal global [7 x i8] c"\0A  %s\0A\00"		; <[7 x i8]*> [#uses=0]
@str111 = internal global [45 x i8] c"\0A     a    b    c    d    e    f    g    h\0A\0A\00"		; <[45 x i8]*> [#uses=0]
@str29.upgrd.167 = internal global [45 x i8] c"\0A     h    g    f    e    d    c    b    a\0A\0A\00"		; <[45 x i8]*> [#uses=0]
@str33.upgrd.168 = internal global [2 x i8] c"<\00"		; <[2 x i8]*> [#uses=0]
@str34.upgrd.169 = internal global [3 x i8] c"> \00"		; <[3 x i8]*> [#uses=0]
@str114.upgrd.170 = internal global [18 x i8] c"%2i %7i %5i %8i  \00"		; <[18 x i8]*> [#uses=0]
@str115 = internal global [20 x i8] c"%2i %c%1i.%02i %9i \00"		; <[20 x i8]*> [#uses=0]
@str39.upgrd.171 = internal global [5 x i8] c"%s !\00"		; <[5 x i8]*> [#uses=0]
@str40.upgrd.172 = internal global [6 x i8] c"%s !!\00"		; <[6 x i8]*> [#uses=0]
@str41.upgrd.173 = internal global [6 x i8] c"%s ??\00"		; <[6 x i8]*> [#uses=0]
@str124 = internal global [71 x i8] c"\0ASjeng version SPEC 1.0, Copyright (C) 2000-2005 Gian-Carlo Pascutto\0A\0A\00"		; <[71 x i8]*> [#uses=0]
@state = internal global [625 x i32] zeroinitializer		; <[625 x i32]*> [#uses=0]

declare fastcc i32 @calc_attackers(i32, i32)

declare fastcc i32 @is_attacked(i32, i32)

declare fastcc void @ProcessHoldings(i8*)

declare void @llvm.memset.i32(i8*, i8, i32, i32)

declare i8* @strncpy(i8*, i8*, i32)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @__eprintf(i8*, i8*, i32, i8*)

declare fastcc void @addHolding(i32, i32)

declare fastcc void @removeHolding(i32, i32)

declare fastcc void @DropremoveHolding(i32, i32)

declare i32 @printf(i8*, ...)

declare fastcc i32 @is_draw()

declare void @exit(i32)

declare fastcc void @setup_epd_line(i8*)

declare i32 @atoi(i8*)

declare fastcc void @reset_piece_square()

declare fastcc void @initialize_hash()

declare i32 @__maskrune(i32, i32)

declare fastcc void @comp_to_san(i64, i64, i64, i8*)

declare i8* @strstr(i8*, i8*)

declare i32 @atol(i8*)

declare %struct.FILE* @fopen(i8*, i8*)

declare fastcc void @display_board(i32)

define internal void @think(%struct.move_s* sret  %agg.result) {
entry:
	%output.i = alloca [8 x i8], align 8		; <[8 x i8]*> [#uses=0]
	%comp_move = alloca %struct.move_s, align 16		; <%struct.move_s*> [#uses=7]
	%temp_move = alloca %struct.move_s, align 16		; <%struct.move_s*> [#uses=6]
	%moves = alloca [512 x %struct.move_s], align 16		; <[512 x %struct.move_s]*> [#uses=7]
	%output = alloca [8 x i8], align 8		; <[8 x i8]*> [#uses=1]
	store i1 false, i1* @userealholdings.b
	%tmp = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0		; <%struct.move_s*> [#uses=3]
	%tmp362 = getelementptr %struct.move_s* %comp_move, i32 0, i32 0		; <i32*> [#uses=0]
	%tmp365 = getelementptr %struct.move_s* %comp_move, i32 0, i32 1		; <i32*> [#uses=0]
	%tmp368 = getelementptr %struct.move_s* %comp_move, i32 0, i32 2		; <i32*> [#uses=0]
	%tmp371 = getelementptr %struct.move_s* %comp_move, i32 0, i32 3		; <i32*> [#uses=0]
	%tmp374 = getelementptr %struct.move_s* %comp_move, i32 0, i32 4		; <i32*> [#uses=0]
	%tmp377 = getelementptr %struct.move_s* %comp_move, i32 0, i32 5		; <i32*> [#uses=0]
	%tmp.upgrd.174 = bitcast %struct.move_s* %comp_move to { i64, i64, i64 }*		; <{ i64, i64, i64 }*> [#uses=3]
	%tmp.upgrd.175 = getelementptr { i64, i64, i64 }* %tmp.upgrd.174, i32 0, i32 0		; <i64*> [#uses=0]
	%tmp829 = getelementptr { i64, i64, i64 }* %tmp.upgrd.174, i32 0, i32 1		; <i64*> [#uses=0]
	%tmp832 = getelementptr { i64, i64, i64 }* %tmp.upgrd.174, i32 0, i32 2		; <i64*> [#uses=0]
	%output.upgrd.176 = getelementptr [8 x i8]* %output, i32 0, i32 0		; <i8*> [#uses=0]
	%tmp573 = getelementptr %struct.move_s* %temp_move, i32 0, i32 0		; <i32*> [#uses=0]
	%tmp576 = getelementptr %struct.move_s* %temp_move, i32 0, i32 1		; <i32*> [#uses=0]
	%tmp579 = getelementptr %struct.move_s* %temp_move, i32 0, i32 2		; <i32*> [#uses=0]
	%tmp582 = getelementptr %struct.move_s* %temp_move, i32 0, i32 3		; <i32*> [#uses=0]
	%tmp585 = getelementptr %struct.move_s* %temp_move, i32 0, i32 4		; <i32*> [#uses=0]
	%tmp588 = getelementptr %struct.move_s* %temp_move, i32 0, i32 5		; <i32*> [#uses=0]
	%pn_restart.0.ph = bitcast i32 0 to i32		; <i32> [#uses=2]
	%tmp21362 = icmp eq i32 0, 0		; <i1> [#uses=2]
	%tmp216 = sitofp i32 %pn_restart.0.ph to float		; <float> [#uses=1]
	%tmp216.upgrd.177 = fpext float %tmp216 to double		; <double> [#uses=1]
	%tmp217 = add double %tmp216.upgrd.177, 1.000000e+00		; <double> [#uses=1]
	%tmp835 = icmp sgt i32 %pn_restart.0.ph, 9		; <i1> [#uses=0]
	store i32 0, i32* @nodes
	store i32 0, i32* @qnodes
	store i32 1, i32* @ply
	store i32 0, i32* @ECacheProbes
	store i32 0, i32* @ECacheHits
	store i32 0, i32* @TTProbes
	store i32 0, i32* @TTHits
	store i32 0, i32* @TTStores
	store i32 0, i32* @NCuts
	store i32 0, i32* @NTries
	store i32 0, i32* @TExt
	store i32 0, i32* @FH
	store i32 0, i32* @FHF
	store i32 0, i32* @PVS
	store i32 0, i32* @FULL
	store i32 0, i32* @PVSF
	store i32 0, i32* @ext_check
	store i32 0, i32* @ext_onerep
	store i32 0, i32* @razor_drop
	store i32 0, i32* @razor_material
	store i1 false, i1* @extendedtime.b
	store i1 false, i1* @forcedwin.b
	store i32 200, i32* @maxposdiff
	store i8 0, i8* @true_i_depth
	store i32 0, i32* @legals
	%tmp48 = load i32* @Variant		; <i32> [#uses=1]
	%tmp49 = icmp eq i32 %tmp48, 4		; <i1> [#uses=1]
	%storemerge = zext i1 %tmp49 to i32		; <i32> [#uses=1]
	store i32 %storemerge, i32* @captures
	call fastcc void @gen( %struct.move_s* %tmp )
	%tmp53 = load i32* @numb_moves		; <i32> [#uses=1]
	%tmp.i = load i32* @Variant		; <i32> [#uses=1]
	%tmp.i.upgrd.178 = icmp eq i32 %tmp.i, 3		; <i1> [#uses=1]
	br i1 %tmp.i.upgrd.178, label %in_check.exit, label %cond_next.i

cond_next.i:		; preds = %entry
	%tmp2.i5 = load i32* @white_to_move		; <i32> [#uses=1]
	%tmp3.i = icmp eq i32 %tmp2.i5, 1		; <i1> [#uses=0]
	ret void

in_check.exit:		; preds = %entry
	%tmp7637 = icmp sgt i32 %tmp53, 0		; <i1> [#uses=1]
	br i1 %tmp7637, label %cond_true77, label %bb80

cond_true77:		; preds = %in_check.exit
	%l.1.0 = bitcast i32 0 to i32		; <i32> [#uses=2]
	call fastcc void @make( %struct.move_s* %tmp, i32 %l.1.0 )
	%tmp61 = call fastcc i32 @check_legal( %struct.move_s* %tmp, i32 %l.1.0, i32 0 )		; <i32> [#uses=1]
	%tmp62 = icmp eq i32 %tmp61, 0		; <i1> [#uses=0]
	ret void

bb80:		; preds = %in_check.exit
	%tmp81 = load i32* @Variant		; <i32> [#uses=1]
	%tmp82 = icmp eq i32 %tmp81, 4		; <i1> [#uses=1]
	br i1 %tmp82, label %cond_true83, label %cond_next118

cond_true83:		; preds = %bb80
	%tmp84 = load i32* @legals		; <i32> [#uses=1]
	%tmp85 = icmp eq i32 %tmp84, 0		; <i1> [#uses=0]
	ret void

cond_next118:		; preds = %bb80
	%tmp119 = load i32* @Variant		; <i32> [#uses=1]
	%tmp120 = icmp eq i32 %tmp119, 1		; <i1> [#uses=1]
	br i1 %tmp120, label %cond_next176, label %cond_true121

cond_true121:		; preds = %cond_next118
	%tmp122.b = load i1* @is_pondering.b		; <i1> [#uses=1]
	br i1 %tmp122.b, label %cond_next176, label %cond_true124

cond_true124:		; preds = %cond_true121
	%tmp125 = load i32* @legals		; <i32> [#uses=1]
	%tmp126 = icmp eq i32 %tmp125, 1		; <i1> [#uses=1]
	br i1 %tmp126, label %cond_true127, label %cond_next176

cond_true127:		; preds = %cond_true124
	%tmp128 = load i32* @inc		; <i32> [#uses=1]
	%tmp129 = mul i32 %tmp128, 100		; <i32> [#uses=1]
	%tmp130 = load i32* @time_cushion		; <i32> [#uses=1]
	%tmp131 = add i32 %tmp129, %tmp130		; <i32> [#uses=1]
	store i32 %tmp131, i32* @time_cushion
	%tmp134 = getelementptr %struct.move_s* %agg.result, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp135 = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp136 = load i32* %tmp135		; <i32> [#uses=1]
	store i32 %tmp136, i32* %tmp134
	%tmp137 = getelementptr %struct.move_s* %agg.result, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp138 = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp139 = load i32* %tmp138		; <i32> [#uses=1]
	store i32 %tmp139, i32* %tmp137
	%tmp140 = getelementptr %struct.move_s* %agg.result, i32 0, i32 2		; <i32*> [#uses=1]
	%tmp141 = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0, i32 2		; <i32*> [#uses=1]
	%tmp142 = load i32* %tmp141		; <i32> [#uses=1]
	store i32 %tmp142, i32* %tmp140
	%tmp143 = getelementptr %struct.move_s* %agg.result, i32 0, i32 3		; <i32*> [#uses=1]
	%tmp144 = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0, i32 3		; <i32*> [#uses=1]
	%tmp145 = load i32* %tmp144		; <i32> [#uses=1]
	store i32 %tmp145, i32* %tmp143
	%tmp146 = getelementptr %struct.move_s* %agg.result, i32 0, i32 4		; <i32*> [#uses=1]
	%tmp147 = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0, i32 4		; <i32*> [#uses=1]
	%tmp148 = load i32* %tmp147		; <i32> [#uses=1]
	store i32 %tmp148, i32* %tmp146
	%tmp149 = getelementptr %struct.move_s* %agg.result, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp150 = getelementptr [512 x %struct.move_s]* %moves, i32 0, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp151 = load i32* %tmp150		; <i32> [#uses=1]
	store i32 %tmp151, i32* %tmp149
	ret void

cond_next176:		; preds = %cond_true124, %cond_true121, %cond_next118
	call fastcc void @check_phase( )
	%tmp177 = load i32* @phase		; <i32> [#uses=1]
	switch i32 %tmp177, label %bb187 [
		 i32 0, label %bb178
		 i32 1, label %bb180
		 i32 2, label %bb183
	]

bb178:		; preds = %cond_next176
	%tmp179 = call i32 (i8*, ...)* @printf( i8* getelementptr ([16 x i8]* @str40.upgrd.84, i32 0, i64 0) )		; <i32> [#uses=0]
	%tmp18854.b = load i1* @is_pondering.b		; <i1> [#uses=1]
	br i1 %tmp18854.b, label %cond_false210, label %cond_true190

bb180:		; preds = %cond_next176
	%tmp182 = call i32 (i8*, ...)* @printf( i8* getelementptr ([19 x i8]* @str.upgrd.85, i32 0, i64 0) )		; <i32> [#uses=0]
	%tmp18856.b = load i1* @is_pondering.b		; <i1> [#uses=0]
	ret void

bb183:		; preds = %cond_next176
	%tmp185 = call i32 (i8*, ...)* @printf( i8* getelementptr ([16 x i8]* @str1.upgrd.86, i32 0, i64 0) )		; <i32> [#uses=0]
	%tmp18858.b = load i1* @is_pondering.b		; <i1> [#uses=0]
	ret void

bb187:		; preds = %cond_next176
	%tmp188.b = load i1* @is_pondering.b		; <i1> [#uses=0]
	ret void

cond_true190:		; preds = %bb178
	%tmp191 = load i32* @fixed_time		; <i32> [#uses=1]
	%tmp192 = icmp eq i32 %tmp191, 0		; <i1> [#uses=0]
	ret void

cond_false210:		; preds = %bb178
	store i32 999999, i32* @time_for_move
	br i1 %tmp21362, label %cond_true226.critedge, label %bb287.critedge

cond_true226.critedge:		; preds = %cond_false210
	%tmp223.c = call i32 (i8*, ...)* @printf( i8* getelementptr ([20 x i8]* @str43, i32 0, i64 0), i32 999999 )		; <i32> [#uses=0]
	%tmp.i.upgrd.179 = load %struct.TType** @DP_TTable		; <%struct.TType*> [#uses=1]
	%tmp.i7.b = load i1* @TTSize.b		; <i1> [#uses=1]
	%tmp1.i = select i1 %tmp.i7.b, i32 60000000, i32 0		; <i32> [#uses=1]
	%tmp.i.sb = getelementptr %struct.TType* %tmp.i.upgrd.179, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memset.i32( i8* %tmp.i.sb, i8 0, i32 %tmp1.i, i32 4 )
	%tmp2.i = load %struct.TType** @AS_TTable		; <%struct.TType*> [#uses=1]
	%tmp3.i8.b = load i1* @TTSize.b		; <i1> [#uses=1]
	%tmp4.i = select i1 %tmp3.i8.b, i32 60000000, i32 0		; <i32> [#uses=1]
	%tmp2.i.upgrd.180 = getelementptr %struct.TType* %tmp2.i, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memset.i32( i8* %tmp2.i.upgrd.180, i8 0, i32 %tmp4.i, i32 4 )
	%tmp.i.QTT = load %struct.QTType** @QS_TTable		; <%struct.QTType*> [#uses=1]
	%tmp5.i9.b = load i1* @TTSize.b		; <i1> [#uses=1]
	%tmp6.i10 = select i1 %tmp5.i9.b, i32 48000000, i32 0		; <i32> [#uses=1]
	%tmp7.i = getelementptr %struct.QTType* %tmp.i.QTT, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memset.i32( i8* %tmp7.i, i8 0, i32 %tmp6.i10, i32 4 )
	%tmp.i.ECache = load %struct.ECacheType** @ECache		; <%struct.ECacheType*> [#uses=1]
	%tmp.i14.b = load i1* @ECacheSize.b		; <i1> [#uses=1]
	%tmp1.i16 = select i1 %tmp.i14.b, i32 12000000, i32 0		; <i32> [#uses=1]
	%tmp.i17 = bitcast %struct.ECacheType* %tmp.i.ECache to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.i32( i8* %tmp.i17, i8 0, i32 %tmp1.i16, i32 4 )
	call void @llvm.memset.i32( i8* bitcast ([300 x i32]* @rootlosers to i8*), i8 0, i32 1200, i32 4 )
	%tmp234.b = load i1* @is_pondering.b		; <i1> [#uses=1]
	br i1 %tmp234.b, label %bb263, label %cond_next238

cond_next238:		; preds = %cond_true226.critedge
	%tmp239 = load i32* @Variant		; <i32> [#uses=2]
	switch i32 %tmp239, label %bb263 [
		 i32 3, label %bb249
		 i32 4, label %bb249
	]

bb249:		; preds = %cond_next238, %cond_next238
	%tmp250 = load i32* @piece_count		; <i32> [#uses=1]
	%tmp251 = icmp sgt i32 %tmp250, 3		; <i1> [#uses=1]
	%tmp240.not = icmp ne i32 %tmp239, 3		; <i1> [#uses=1]
	%brmerge = or i1 %tmp251, %tmp240.not		; <i1> [#uses=1]
	br i1 %brmerge, label %bb260, label %bb263

bb260:		; preds = %bb249
	%tmp261 = load i32* @time_for_move		; <i32> [#uses=1]
	%tmp261.upgrd.181 = sitofp i32 %tmp261 to float		; <float> [#uses=1]
	%tmp261.upgrd.182 = fpext float %tmp261.upgrd.181 to double		; <double> [#uses=1]
	%tmp262 = fdiv double %tmp261.upgrd.182, 3.000000e+00		; <double> [#uses=1]
	%tmp262.upgrd.183 = fptosi double %tmp262 to i32		; <i32> [#uses=1]
	store i32 %tmp262.upgrd.183, i32* @pn_time
	%tmp1.b.i = load i1* @PBSize.b		; <i1> [#uses=1]
	%tmp1.i1 = select i1 %tmp1.b.i, i32 200000, i32 0		; <i32> [#uses=1]
	%tmp.i2 = call i8* @calloc( i32 %tmp1.i1, i32 44 )		; <i8*> [#uses=1]
	%tmp.i.ub = bitcast i8* %tmp.i2 to i8*		; <i8*> [#uses=1]
	store i8* %tmp.i.ub, i8** @membuff
	%tmp2.i3 = call i8* @calloc( i32 1, i32 44 )		; <i8*> [#uses=3]
	%tmp2.i.upgrd.184 = bitcast i8* %tmp2.i3 to %struct.node_t*		; <%struct.node_t*> [#uses=6]
	%tmp.i.move_s = getelementptr [512 x %struct.move_s]* null, i32 0, i32 0		; <%struct.move_s*> [#uses=3]
	call fastcc void @gen( %struct.move_s* %tmp.i.move_s )
	%tmp3.i4 = load i32* @numb_moves		; <i32> [#uses=4]
	%tmp3.i5 = bitcast i32 %tmp3.i4 to i32		; <i32> [#uses=0]
	store i1 false, i1* @alllosers.b
	call void @llvm.memset.i32( i8* bitcast ([300 x i32]* @rootlosers to i8*), i8 0, i32 1200, i32 4 )
	%nodesspent.i = bitcast [512 x i32]* null to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.i32( i8* %nodesspent.i, i8 0, i32 2048, i32 16 )
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 0)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 1)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 2)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 3)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 4)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 5)
	%tmp.i.i = load i32* @Variant		; <i32> [#uses=1]
	%tmp.i.i.upgrd.185 = icmp eq i32 %tmp.i.i, 3		; <i1> [#uses=1]
	br i1 %tmp.i.i.upgrd.185, label %in_check.exit.i, label %cond_next.i.i

cond_next.i.i:		; preds = %bb260
	%tmp2.i.i = load i32* @white_to_move		; <i32> [#uses=1]
	%tmp3.i.i = icmp eq i32 %tmp2.i.i, 1		; <i1> [#uses=1]
	br i1 %tmp3.i.i, label %cond_true4.i.i, label %cond_false12.i.i

cond_true4.i.i:		; preds = %cond_next.i.i
	%tmp5.i.i = load i32* @wking_loc		; <i32> [#uses=1]
	%tmp6.i.i = call fastcc i32 @is_attacked( i32 %tmp5.i.i, i32 0 )		; <i32> [#uses=1]
	%not.tmp7.i.i = icmp ne i32 %tmp6.i.i, 0		; <i1> [#uses=1]
	%tmp217.i = zext i1 %not.tmp7.i.i to i32		; <i32> [#uses=1]
	%tmp4219.i = icmp sgt i32 %tmp3.i4, 0		; <i1> [#uses=1]
	br i1 %tmp4219.i, label %cond_true43.i, label %bb46.i

cond_false12.i.i:		; preds = %cond_next.i.i
	%tmp13.i.i = load i32* @bking_loc		; <i32> [#uses=1]
	%tmp14.i.i = call fastcc i32 @is_attacked( i32 %tmp13.i.i, i32 1 )		; <i32> [#uses=1]
	%not.tmp15.i.i = icmp ne i32 %tmp14.i.i, 0		; <i1> [#uses=1]
	%tmp2120.i = zext i1 %not.tmp15.i.i to i32		; <i32> [#uses=1]
	%tmp4222.i = icmp sgt i32 %tmp3.i4, 0		; <i1> [#uses=1]
	br i1 %tmp4222.i, label %cond_true43.i, label %bb46.i

in_check.exit.i:		; preds = %bb260
	%tmp4224.i = icmp sgt i32 %tmp3.i4, 0		; <i1> [#uses=0]
	ret void

cond_true43.i:		; preds = %cond_false12.i.i, %cond_true4.i.i
	%tmp21.0.ph.i = phi i32 [ %tmp217.i, %cond_true4.i.i ], [ %tmp2120.i, %cond_false12.i.i ]		; <i32> [#uses=1]
	%i.0.0.i = bitcast i32 0 to i32		; <i32> [#uses=2]
	call fastcc void @make( %struct.move_s* %tmp.i.move_s, i32 %i.0.0.i )
	%tmp27.i = call fastcc i32 @check_legal( %struct.move_s* %tmp.i.move_s, i32 %i.0.0.i, i32 %tmp21.0.ph.i )		; <i32> [#uses=1]
	%tmp.i6 = icmp eq i32 %tmp27.i, 0		; <i1> [#uses=0]
	ret void

bb46.i:		; preds = %cond_false12.i.i, %cond_true4.i.i
	%tmp48.i = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp48.i, label %cond_true49.i, label %cond_next53.i

cond_true49.i:		; preds = %bb46.i
	store i32 0, i32* @bufftop
	%tmp50.i = load i8** @membuff		; <i8*> [#uses=1]
	free i8* %tmp50.i
	free i8* %tmp2.i3
	ret void

cond_next53.i:		; preds = %bb46.i
	store i32 1, i32* @nodecount
	store i32 0, i32* @iters
	store i32 0, i32* @maxply
	store i32 0, i32* @forwards
	%tmp54.i = load i32* @move_number		; <i32> [#uses=1]
	%tmp55.i = load i32* @ply		; <i32> [#uses=1]
	%tmp56.i = add i32 %tmp54.i, -1		; <i32> [#uses=1]
	%tmp57.i = add i32 %tmp56.i, %tmp55.i		; <i32> [#uses=1]
	%tmp58.i = load i32* @hash		; <i32> [#uses=1]
	%tmp.i.upgrd.186 = getelementptr [600 x i32]* @hash_history, i32 0, i32 %tmp57.i		; <i32*> [#uses=1]
	store i32 %tmp58.i, i32* %tmp.i.upgrd.186
	%tmp59.i = load i32* @white_to_move		; <i32> [#uses=1]
	%tmp60.i = icmp eq i32 %tmp59.i, 0		; <i1> [#uses=1]
	%tmp60.i.upgrd.187 = zext i1 %tmp60.i to i32		; <i32> [#uses=1]
	store i32 %tmp60.i.upgrd.187, i32* @root_to_move
	%tmp.i4.i = load i32* @Variant		; <i32> [#uses=2]
	%tmp.i5.i = icmp eq i32 %tmp.i4.i, 3		; <i1> [#uses=1]
	br i1 %tmp.i5.i, label %cond_true.i.i, label %cond_false.i.i

cond_true.i.i:		; preds = %cond_next53.i
	call fastcc void @suicide_pn_eval( %struct.node_t* %tmp2.i.upgrd.184 )
	%tmp6328.i = getelementptr %struct.node_t* %tmp2.i.upgrd.184, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp29.i = load i8* %tmp6328.i		; <i8> [#uses=1]
	%tmp6430.i = icmp eq i8 %tmp29.i, 1		; <i1> [#uses=0]
	ret void

cond_false.i.i:		; preds = %cond_next53.i
	%tmp2.i.i.upgrd.188 = icmp eq i32 %tmp.i4.i, 4		; <i1> [#uses=1]
	%tmp63.i = getelementptr %struct.node_t* %tmp2.i.upgrd.184, i32 0, i32 0		; <i8*> [#uses=2]
	br i1 %tmp2.i.i.upgrd.188, label %cond_true3.i.i, label %cond_false5.i.i

cond_true3.i.i:		; preds = %cond_false.i.i
	call fastcc void @losers_pn_eval( %struct.node_t* %tmp2.i.upgrd.184 )
	%tmp31.i = load i8* %tmp63.i		; <i8> [#uses=1]
	%tmp6432.i = icmp eq i8 %tmp31.i, 1		; <i1> [#uses=1]
	br i1 %tmp6432.i, label %bb75.i, label %cond_next67.i

cond_false5.i.i:		; preds = %cond_false.i.i
	call fastcc void @std_pn_eval( %struct.node_t* %tmp2.i.upgrd.184 )
	%tmp.i.upgrd.189 = load i8* %tmp63.i		; <i8> [#uses=1]
	%tmp64.i = icmp eq i8 %tmp.i.upgrd.189, 1		; <i1> [#uses=0]
	ret void

cond_next67.i:		; preds = %cond_true3.i.i
	%tmp69.i = getelementptr %struct.node_t* %tmp2.i.upgrd.184, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp70.i = load i8* %tmp69.i		; <i8> [#uses=1]
	%tmp71.i = icmp eq i8 %tmp70.i, 0		; <i1> [#uses=0]
	ret void

bb75.i:		; preds = %cond_true3.i.i
	store i32 0, i32* @bufftop
	%tmp76.i = load i8** @membuff		; <i8*> [#uses=1]
	free i8* %tmp76.i
	free i8* %tmp2.i3
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 0)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 1)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 2)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 3)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 4)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 5)
	%tmp28869 = load i32* @result		; <i32> [#uses=1]
	%tmp28970 = icmp eq i32 %tmp28869, 0		; <i1> [#uses=1]
	br i1 %tmp28970, label %cond_next337, label %cond_true290

bb263:		; preds = %bb249, %cond_next238, %cond_true226.critedge
	br i1 %tmp21362, label %cond_true266, label %bb287

cond_true266:		; preds = %bb263
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 0)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 1)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 2)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 3)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 4)
	store i32 0, i32* getelementptr (%struct.move_s* @pn_move, i64 0, i32 5)
	%tmp28871 = load i32* @result		; <i32> [#uses=1]
	%tmp28972 = icmp eq i32 %tmp28871, 0		; <i1> [#uses=0]
	ret void

bb287.critedge:		; preds = %cond_false210
	%tmp218.c = fdiv double 1.999998e+06, %tmp217		; <double> [#uses=1]
	%tmp218.c.upgrd.190 = fptosi double %tmp218.c to i32		; <i32> [#uses=2]
	store i32 %tmp218.c.upgrd.190, i32* @time_for_move
	%tmp22367.c = call i32 (i8*, ...)* @printf( i8* getelementptr ([20 x i8]* @str43, i32 0, i64 0), i32 %tmp218.c.upgrd.190 )		; <i32> [#uses=0]
	ret void

bb287:		; preds = %bb263
	%tmp288 = load i32* @result		; <i32> [#uses=1]
	%tmp289 = icmp eq i32 %tmp288, 0		; <i1> [#uses=0]
	ret void

cond_true290:		; preds = %bb75.i
	%tmp292 = load i32* getelementptr (%struct.move_s* @pn_move, i32 0, i32 1)		; <i32> [#uses=1]
	%tmp295 = icmp eq i32 %tmp292, 0		; <i1> [#uses=0]
	ret void

cond_next337:		; preds = %bb75.i
	%tmp338.b = load i1* @forcedwin.b		; <i1> [#uses=1]
	br i1 %tmp338.b, label %bb348, label %cond_next342

cond_next342:		; preds = %cond_next337
	%tmp343 = load i32* @result		; <i32> [#uses=1]
	%tmp344 = icmp eq i32 %tmp343, 0		; <i1> [#uses=0]
	ret void

bb348:		; preds = %cond_next337
	%tmp350 = load i32* getelementptr (%struct.move_s* @pn_move, i32 0, i32 1)		; <i32> [#uses=1]
	%tmp353 = icmp eq i32 %tmp350, 0		; <i1> [#uses=0]
	ret void
}

declare fastcc i32 @eval(i32, i32)

declare i8* @fgets(i8*, i32, %struct.FILE*)

declare i32 @fclose(%struct.FILE*)

declare fastcc i32 @losers_eval()

declare fastcc i32 @l_bishop_mobility(i32)

declare fastcc i32 @l_rook_mobility(i32)

declare fastcc i32 @check_legal(%struct.move_s*, i32, i32)

declare fastcc void @gen(%struct.move_s*)

declare fastcc void @push_pawn(i32, i32)

declare fastcc void @push_knighT(i32)

declare fastcc void @push_slidE(i32)

declare fastcc void @push_king(i32)

declare fastcc i32 @f_in_check(%struct.move_s*, i32)

declare fastcc void @make(%struct.move_s*, i32)

declare fastcc void @add_capture(i32, i32, i32)

declare fastcc void @unmake(%struct.move_s*, i32)

declare i32 @ErrorIt(i32, i32)

declare i32 @Pawn(i32, i32)

declare i32 @Knight(i32, i32)

declare i32 @King(i32, i32)

declare i32 @Rook(i32, i32)

declare i32 @Queen(i32, i32)

declare i32 @Bishop(i32, i32)

declare fastcc void @check_phase()

declare fastcc i32 @bishop_mobility(i32)

declare fastcc i32 @rook_mobility(i32)

declare i32 @sscanf(i8*, i8*, ...)

declare i32 @strncmp(i8*, i8*, i32)

declare i8* @strchr(i8*, i32)

declare fastcc void @CheckBadFlow(i32)

declare fastcc void @suicide_pn_eval(%struct.node_t*)

declare fastcc void @losers_pn_eval(%struct.node_t*)

declare fastcc void @std_pn_eval(%struct.node_t*)

declare fastcc %struct.node_t* @select_most_proving(%struct.node_t*)

declare fastcc void @set_proof_and_disproof_numbers(%struct.node_t*)

declare fastcc void @StoreTT(i32, i32, i32, i32, i32, i32)

declare fastcc void @develop_node(%struct.node_t*)

declare fastcc void @update_ancestors(%struct.node_t*)

declare i8* @calloc(i32, i32)

declare fastcc void @comp_to_coord(i64, i64, i64, i8*)

declare i8* @strcat(i8*, i8*)

declare i32 @sprintf(i8*, i8*, ...)

declare fastcc void @order_moves(%struct.move_s*, i32*, i32*, i32, i32)

declare fastcc i32 @see(i32, i32, i32)

declare fastcc void @perft(i32)

declare fastcc i32 @qsearch(i32, i32, i32)

declare fastcc i32 @allocate_time()

declare fastcc void @QStoreTT(i32, i32, i32, i32)

declare fastcc i32 @search(i32, i32, i32, i32)

declare fastcc i32 @ProbeTT(i32*, i32, i32*, i32*, i32*, i32)

declare void @search_root(%struct.move_s* sret , i32, i32, i32)

declare fastcc void @post_fh_thinking(i32, %struct.move_s*)

declare fastcc void @post_thinking(i32)

declare i32 @fprintf(%struct.FILE*, i8*, ...)

declare fastcc i32 @s_bishop_mobility(i32)

declare fastcc i32 @s_rook_mobility(i32)

declare fastcc i32 @suicide_mid_eval()

declare i32 @main(i32, i8**)

declare fastcc void @init_game()

declare void @setbuf(%struct.FILE*, i8*)

declare i8* @strcpy(i8*, i8*)

declare i32 @__tolower(i32)

declare i32 @strcmp(i8*, i8*)

declare void (i32)* @signal(i32, void (i32)*)

declare fastcc void @hash_extract_pv(i32, i8*)

declare double @difftime(i32, i32)

declare i32 @getc(%struct.FILE*)

declare i32 @strlen(i8*)

declare i32 @fwrite(i8*, i32, i32, %struct.FILE*)
