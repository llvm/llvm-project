; RUN: llvm-as < %s | opt -loop-unswitch -disable-output
; END.

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.2.0"
deplibs = [ "c", "crtend" ]
	%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.fd_set = type { [32 x i32] }
	%struct.timeval = type { i32, i32 }
	%struct.tm = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8* }
	%typedef.CHESS_PATH = type { [65 x i32], i8, i8, i8 }
	%typedef.CHESS_POSITION = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i8, i8, [64 x i8], i8, i8, i8, i8, i8 }
	%typedef.HASH_ENTRY = type { i64, i64 }
	%typedef.NEXT_MOVE = type { i32, i32, i32* }
	%typedef.PAWN_HASH_ENTRY = type { i32, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%typedef.SEARCH_POSITION = type { i8, i8, i8, i8 }
	%union.doub0. = type { i64 }
@search = external global %typedef.CHESS_POSITION		; <%typedef.CHESS_POSITION*> [#uses=1]
@w_pawn_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_pawn_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@knight_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@bishop_attacks_rl45 = external global [64 x [256 x i64]]		; <[64 x [256 x i64]]*> [#uses=0]
@bishop_shift_rl45 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@bishop_attacks_rr45 = external global [64 x [256 x i64]]		; <[64 x [256 x i64]]*> [#uses=0]
@bishop_shift_rr45 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@rook_attacks_r0 = external global [64 x [256 x i64]]		; <[64 x [256 x i64]]*> [#uses=0]
@rook_attacks_rl90 = external global [64 x [256 x i64]]		; <[64 x [256 x i64]]*> [#uses=0]
@king_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@set_mask = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@first_ones = external global [65536 x i8]		; <[65536 x i8]*> [#uses=0]
@last_ones = external global [65536 x i8]		; <[65536 x i8]*> [#uses=0]
@draw_score_is_zero = external global i32		; <i32*> [#uses=0]
@default_draw_score = external global i32		; <i32*> [#uses=0]
@opening = external global i32		; <i32*> [#uses=0]
@middle_game = external global i32		; <i32*> [#uses=0]
@tc_increment = external global i32		; <i32*> [#uses=0]
@tc_time_remaining_opponent = external global i32		; <i32*> [#uses=0]
@.ctor_1 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@input_stream = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@__sF = external global [0 x %struct.__sFILE]		; <[0 x %struct.__sFILE]*> [#uses=1]
@xboard = external global i32		; <i32*> [#uses=0]
@.str_1 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_2 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@buffer = external global [512 x i8]		; <[512 x i8]*> [#uses=0]
@nargs = external global i32		; <i32*> [#uses=0]
@args = external global [32 x i8*]		; <[32 x i8*]*> [#uses=0]
@.str_3 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_4 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_5 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_6 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_7 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_8 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_9 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_10 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_11 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_12 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_14 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@position = external global [67 x %typedef.SEARCH_POSITION]		; <[67 x %typedef.SEARCH_POSITION]*> [#uses=0]
@log_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@move_number = external global i32		; <i32*> [#uses=0]
@rephead_b = external global i64*		; <i64**> [#uses=0]
@replist_b = external global [82 x i64]		; <[82 x i64]*> [#uses=0]
@rephead_w = external global i64*		; <i64**> [#uses=0]
@replist_w = external global [82 x i64]		; <[82 x i64]*> [#uses=0]
@moves_out_of_book = external global i32		; <i32*> [#uses=0]
@largest_positional_score = external global i32		; <i32*> [#uses=0]
@end_game = external global i32		; <i32*> [#uses=0]
@p_values = external global [15 x i32]		; <[15 x i32]*> [#uses=0]
@clear_mask = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@directions = external global [64 x [64 x i8]]		; <[64 x [64 x i8]]*> [#uses=0]
@root_wtm = external global i32		; <i32*> [#uses=0]
@all_pawns = external global i64		; <i64*> [#uses=0]
@pawn_score = external global %typedef.PAWN_HASH_ENTRY		; <%typedef.PAWN_HASH_ENTRY*> [#uses=0]
@pawn_probes = external global i32		; <i32*> [#uses=0]
@pawn_hits = external global i32		; <i32*> [#uses=0]
@outside_passed = external global [128 x i32]		; <[128 x i32]*> [#uses=0]
@root_total_black_pieces = external global i32		; <i32*> [#uses=0]
@root_total_white_pawns = external global i32		; <i32*> [#uses=0]
@root_total_white_pieces = external global i32		; <i32*> [#uses=0]
@root_total_black_pawns = external global i32		; <i32*> [#uses=0]
@mask_A7H7 = external global i64		; <i64*> [#uses=0]
@mask_B6B7 = external global i64		; <i64*> [#uses=0]
@mask_G6G7 = external global i64		; <i64*> [#uses=0]
@mask_A2H2 = external global i64		; <i64*> [#uses=0]
@mask_B2B3 = external global i64		; <i64*> [#uses=0]
@mask_G2G3 = external global i64		; <i64*> [#uses=0]
@king_defects_w = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@good_bishop_kw = external global i64		; <i64*> [#uses=0]
@mask_F3H3 = external global i64		; <i64*> [#uses=0]
@file_mask = external global [8 x i64]		; <[8 x i64]*> [#uses=0]
@good_bishop_qw = external global i64		; <i64*> [#uses=0]
@mask_A3C3 = external global i64		; <i64*> [#uses=0]
@king_defects_b = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@good_bishop_kb = external global i64		; <i64*> [#uses=0]
@mask_F6H6 = external global i64		; <i64*> [#uses=0]
@good_bishop_qb = external global i64		; <i64*> [#uses=0]
@mask_A6C6 = external global i64		; <i64*> [#uses=0]
@square_color = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@evaluations = external global i32		; <i32*> [#uses=0]
@king_value_w = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@rank_mask = external global [8 x i64]		; <[8 x i64]*> [#uses=0]
@mask_kr_trapped_w = external global [3 x i64]		; <[3 x i64]*> [#uses=0]
@mask_qr_trapped_w = external global [3 x i64]		; <[3 x i64]*> [#uses=0]
@king_value_b = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@mask_kr_trapped_b = external global [3 x i64]		; <[3 x i64]*> [#uses=0]
@mask_qr_trapped_b = external global [3 x i64]		; <[3 x i64]*> [#uses=0]
@white_outpost = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@mask_no_pawn_attacks_b = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@knight_value_w = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@black_outpost = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@mask_no_pawn_attacks_w = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@knight_value_b = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@bishop_value_w = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@bishop_mobility_rl45 = external global [64 x [256 x i32]]		; <[64 x [256 x i32]]*> [#uses=0]
@bishop_mobility_rr45 = external global [64 x [256 x i32]]		; <[64 x [256 x i32]]*> [#uses=0]
@bishop_value_b = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@rook_value_w = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@plus8dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@mask_abs7_w = external global i64		; <i64*> [#uses=0]
@rook_value_b = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@minus8dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@mask_abs7_b = external global i64		; <i64*> [#uses=0]
@queen_value_w = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@queen_value_b = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@white_minor_pieces = external global i64		; <i64*> [#uses=0]
@black_minor_pieces = external global i64		; <i64*> [#uses=0]
@not_rook_pawns = external global i64		; <i64*> [#uses=0]
@dark_squares = external global i64		; <i64*> [#uses=0]
@b_n_mate_dark_squares = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@b_n_mate_light_squares = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@mate = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@first_ones_8bit = external global [256 x i8]		; <[256 x i8]*> [#uses=0]
@reduced_material_passer = external global [20 x i32]		; <[20 x i32]*> [#uses=0]
@supported_passer = external global [8 x i32]		; <[8 x i32]*> [#uses=0]
@passed_pawn_value = external global [8 x i32]		; <[8 x i32]*> [#uses=0]
@connected_passed = external global [256 x i8]		; <[256 x i8]*> [#uses=0]
@black_pawn_race_btm = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@white_pawn_race_wtm = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@black_pawn_race_wtm = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@white_pawn_race_btm = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@obstructed = external global [64 x [64 x i64]]		; <[64 x [64 x i64]]*> [#uses=0]
@pawn_hash_table = external global %typedef.PAWN_HASH_ENTRY*		; <%typedef.PAWN_HASH_ENTRY**> [#uses=0]
@pawn_hash_mask = external global i32		; <i32*> [#uses=0]
@pawn_value_w = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@mask_pawn_isolated = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@mask_pawn_passed_w = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@mask_pawn_protected_w = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@pawn_value_b = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@mask_pawn_passed_b = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@mask_pawn_protected_b = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@unblocked_pawns = external global [9 x i32]		; <[9 x i32]*> [#uses=0]
@mask_wk_4th = external global i64		; <i64*> [#uses=0]
@mask_wk_5th = external global i64		; <i64*> [#uses=0]
@mask_wq_4th = external global i64		; <i64*> [#uses=0]
@mask_wq_5th = external global i64		; <i64*> [#uses=0]
@stonewall_white = external global i64		; <i64*> [#uses=0]
@mask_bk_4th = external global i64		; <i64*> [#uses=0]
@mask_bk_5th = external global i64		; <i64*> [#uses=0]
@mask_bq_5th = external global i64		; <i64*> [#uses=0]
@mask_bq_4th = external global i64		; <i64*> [#uses=0]
@stonewall_black = external global i64		; <i64*> [#uses=0]
@last_ones_8bit = external global [256 x i8]		; <[256 x i8]*> [#uses=0]
@right_side_mask = external global [8 x i64]		; <[8 x i64]*> [#uses=0]
@left_side_empty_mask = external global [8 x i64]		; <[8 x i64]*> [#uses=0]
@left_side_mask = external global [8 x i64]		; <[8 x i64]*> [#uses=0]
@right_side_empty_mask = external global [8 x i64]		; <[8 x i64]*> [#uses=0]
@pv = external global [65 x %typedef.CHESS_PATH]		; <[65 x %typedef.CHESS_PATH]*> [#uses=0]
@history_w = external global [4096 x i32]		; <[4096 x i32]*> [#uses=0]
@history_b = external global [4096 x i32]		; <[4096 x i32]*> [#uses=0]
@killer_move1 = external global [65 x i32]		; <[65 x i32]*> [#uses=0]
@killer_count1 = external global [65 x i32]		; <[65 x i32]*> [#uses=0]
@killer_move2 = external global [65 x i32]		; <[65 x i32]*> [#uses=0]
@killer_count2 = external global [65 x i32]		; <[65 x i32]*> [#uses=0]
@current_move = external global [65 x i32]		; <[65 x i32]*> [#uses=0]
@init_r90 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@init_l90 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@init_l45 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@init_ul45 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@init_r45 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@init_ur45 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@diagonal_length = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@last = external global [65 x i32*]		; <[65 x i32*]*> [#uses=0]
@move_list = external global [5120 x i32]		; <[5120 x i32]*> [#uses=0]
@history_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@.str_1.upgrd.1 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_2.upgrd.2 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_3.upgrd.3 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_5.upgrd.4 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_6.upgrd.5 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@trans_ref_wa = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
@hash_table_size = external global i32		; <i32*> [#uses=0]
@trans_ref_wb = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
@trans_ref_ba = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
@trans_ref_bb = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
@pawn_hash_table_size = external global i32		; <i32*> [#uses=0]
@.str_9.upgrd.6 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@log_hash = external global i32		; <i32*> [#uses=0]
@log_pawn_hash = external global i32		; <i32*> [#uses=0]
@hash_maska = external global i32		; <i32*> [#uses=0]
@hash_maskb = external global i32		; <i32*> [#uses=0]
@mask_1 = external global i64		; <i64*> [#uses=0]
@bishop_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@queen_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@plus7dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@plus9dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@minus7dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@minus9dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@plus1dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@minus1dir = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@rook_attacks = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@king_attacks_1 = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@king_attacks_2 = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@.ctor_1.upgrd.7 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@.ctor_2 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@rook_mobility_r0 = external global [64 x [256 x i32]]		; <[64 x [256 x i32]]*> [#uses=0]
@rook_mobility_rl90 = external global [64 x [256 x i32]]		; <[64 x [256 x i32]]*> [#uses=0]
@initial_position = external global [80 x i8]		; <[80 x i8]*> [#uses=5]
@"\01a1.0__" = external global [80 x i8]		; <[80 x i8]*> [#uses=0]
@"\01a2.1__" = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@"\01a3.2__" = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@"\01a4.3__" = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@"\01a5.4__" = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@"\01args.5__" = external global [16 x i8*]		; <[16 x i8*]*> [#uses=0]
@.str_10.upgrd.8 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@w_pawn_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@w_pawn_random32 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@b_pawn_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_pawn_random32 = external global [64 x i32]		; <[64 x i32]*> [#uses=0]
@w_knight_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_knight_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@w_bishop_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_bishop_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@w_rook_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_rook_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@w_queen_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_queen_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@w_king_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@b_king_random = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@enpassant_random = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@castle_random_w = external global [2 x i64]		; <[2 x i64]*> [#uses=0]
@castle_random_b = external global [2 x i64]		; <[2 x i64]*> [#uses=0]
@set_mask_rl90 = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@set_mask_rl45 = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@set_mask_rr45 = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@transposition_id = external global i8		; <i8*> [#uses=0]
@mask_2 = external global i64		; <i64*> [#uses=0]
@mask_3 = external global i64		; <i64*> [#uses=0]
@mask_4 = external global i64		; <i64*> [#uses=0]
@mask_8 = external global i64		; <i64*> [#uses=0]
@mask_16 = external global i64		; <i64*> [#uses=0]
@mask_32 = external global i64		; <i64*> [#uses=0]
@mask_72 = external global i64		; <i64*> [#uses=0]
@mask_80 = external global i64		; <i64*> [#uses=0]
@mask_85 = external global i64		; <i64*> [#uses=0]
@mask_96 = external global i64		; <i64*> [#uses=0]
@mask_107 = external global i64		; <i64*> [#uses=0]
@mask_108 = external global i64		; <i64*> [#uses=0]
@mask_112 = external global i64		; <i64*> [#uses=0]
@mask_118 = external global i64		; <i64*> [#uses=0]
@mask_120 = external global i64		; <i64*> [#uses=0]
@mask_121 = external global i64		; <i64*> [#uses=0]
@mask_127 = external global i64		; <i64*> [#uses=0]
@mask_clear_entry = external global i64		; <i64*> [#uses=0]
@clear_mask_rl45 = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@clear_mask_rr45 = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@clear_mask_rl90 = external global [65 x i64]		; <[65 x i64]*> [#uses=0]
@right_half_mask = external global i64		; <i64*> [#uses=0]
@left_half_mask = external global i64		; <i64*> [#uses=0]
@mask_not_rank8 = external global i64		; <i64*> [#uses=0]
@mask_not_rank1 = external global i64		; <i64*> [#uses=0]
@center = external global i64		; <i64*> [#uses=0]
@mask_pawn_connected = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@mask_eptest = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@mask_kingside_attack_w1 = external global i64		; <i64*> [#uses=0]
@mask_kingside_attack_w2 = external global i64		; <i64*> [#uses=0]
@mask_queenside_attack_w1 = external global i64		; <i64*> [#uses=0]
@mask_queenside_attack_w2 = external global i64		; <i64*> [#uses=0]
@mask_kingside_attack_b1 = external global i64		; <i64*> [#uses=0]
@mask_kingside_attack_b2 = external global i64		; <i64*> [#uses=0]
@mask_queenside_attack_b1 = external global i64		; <i64*> [#uses=0]
@mask_queenside_attack_b2 = external global i64		; <i64*> [#uses=0]
@pawns_cramp_black = external global i64		; <i64*> [#uses=0]
@pawns_cramp_white = external global i64		; <i64*> [#uses=0]
@light_squares = external global i64		; <i64*> [#uses=0]
@mask_left_edge = external global i64		; <i64*> [#uses=0]
@mask_right_edge = external global i64		; <i64*> [#uses=0]
@mask_advance_2_w = external global i64		; <i64*> [#uses=0]
@mask_advance_2_b = external global i64		; <i64*> [#uses=0]
@mask_corner_squares = external global i64		; <i64*> [#uses=0]
@mask_promotion_threat_w = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@mask_promotion_threat_b = external global [64 x i64]		; <[64 x i64]*> [#uses=0]
@promote_mask_w = external global i64		; <i64*> [#uses=0]
@promote_mask_b = external global i64		; <i64*> [#uses=0]
@mask_a1_corner = external global i64		; <i64*> [#uses=0]
@mask_h1_corner = external global i64		; <i64*> [#uses=0]
@mask_a8_corner = external global i64		; <i64*> [#uses=0]
@mask_h8_corner = external global i64		; <i64*> [#uses=0]
@white_center_pawns = external global i64		; <i64*> [#uses=0]
@black_center_pawns = external global i64		; <i64*> [#uses=0]
@wtm_random = external global [2 x i64]		; <[2 x i64]*> [#uses=0]
@endgame_random_w = external global i64		; <i64*> [#uses=0]
@endgame_random_b = external global i64		; <i64*> [#uses=0]
@w_rooks_random = external global i64		; <i64*> [#uses=0]
@b_rooks_random = external global i64		; <i64*> [#uses=0]
@.ctor_11 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.ctor_2.upgrd.9 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_1.upgrd.10 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_2.upgrd.11 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_32 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_4.upgrd.12 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_5.upgrd.13 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_6.upgrd.14 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_7.upgrd.15 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_8.upgrd.16 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_9.upgrd.17 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_10.upgrd.18 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_11.upgrd.19 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_12.upgrd.20 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_13 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@num_ponder_moves = external global i32		; <i32*> [#uses=0]
@ponder_moves = external global [220 x i32]		; <[220 x i32]*> [#uses=0]
@.str_14.upgrd.21 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_15 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_16 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@auto232 = external global i32		; <i32*> [#uses=0]
@puzzling = external global i8		; <i8*> [#uses=0]
@abort_search = external global i8		; <i8*> [#uses=0]
@.str_24 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@wtm = external global i32		; <i32*> [#uses=0]
@.str_3.upgrd.22 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_4.upgrd.23 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@end_time = external global i32		; <i32*> [#uses=0]
@time_type = external global i32		; <i32*> [#uses=0]
@start_time = external global i32		; <i32*> [#uses=0]
@.str_6.upgrd.24 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_7.upgrd.25 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@nodes_searched = external global i32		; <i32*> [#uses=0]
@iteration_depth = external global i32		; <i32*> [#uses=0]
@searched_this_root_move = external global [256 x i8]		; <[256 x i8]*> [#uses=0]
@.str_9.upgrd.26 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_10.upgrd.27 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_11.upgrd.28 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_12.upgrd.29 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_14.upgrd.30 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_16.upgrd.31 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@thinking = external global i8		; <i8*> [#uses=0]
@time_abort = external global i32		; <i32*> [#uses=0]
@.str_17 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@analyze_move_read = external global i32		; <i32*> [#uses=0]
@analyze_mode = external global i32		; <i32*> [#uses=0]
@pondering = external global i8		; <i8*> [#uses=0]
@auto232_delay = external global i32		; <i32*> [#uses=0]
@auto_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@.str_19 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_20 = external global [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str_21 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@ponder_move = external global i32		; <i32*> [#uses=0]
@predicted = external global i32		; <i32*> [#uses=0]
@made_predicted_move = external global i32		; <i32*> [#uses=0]
@opponent_end_time = external global i32		; <i32*> [#uses=0]
@program_start_time = external global i32		; <i32*> [#uses=0]
@.str_23 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_24.upgrd.32 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_25 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_26 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_28 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@book_move = external global i32		; <i32*> [#uses=0]
@elapsed_start = external global i32		; <i32*> [#uses=0]
@burp = external global i32		; <i32*> [#uses=0]
@cpu_percent = external global i32		; <i32*> [#uses=0]
@next_time_check = external global i32		; <i32*> [#uses=0]
@nodes_between_time_checks = external global i32		; <i32*> [#uses=0]
@transposition_hits = external global i32		; <i32*> [#uses=0]
@transposition_probes = external global i32		; <i32*> [#uses=0]
@tb_probes = external global i32		; <i32*> [#uses=0]
@tb_probes_successful = external global i32		; <i32*> [#uses=0]
@check_extensions_done = external global i32		; <i32*> [#uses=0]
@recapture_extensions_done = external global i32		; <i32*> [#uses=0]
@passed_pawn_extensions_done = external global i32		; <i32*> [#uses=0]
@one_reply_extensions_done = external global i32		; <i32*> [#uses=0]
@program_end_time = external global i32		; <i32*> [#uses=0]
@root_value = external global i32		; <i32*> [#uses=0]
@last_search_value = external global i32		; <i32*> [#uses=0]
@.str_1.upgrd.33 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_2.upgrd.34 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@booking = external global i8		; <i8*> [#uses=0]
@annotate_mode = external global i32		; <i32*> [#uses=0]
@.str_4.upgrd.35 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_5.upgrd.36 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@last_pv = external global %typedef.CHESS_PATH		; <%typedef.CHESS_PATH*> [#uses=0]
@.str_8.upgrd.37 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@root_alpha = external global i32		; <i32*> [#uses=0]
@last_value = external global i32		; <i32*> [#uses=0]
@root_beta = external global i32		; <i32*> [#uses=0]
@root_nodes = external global [256 x i32]		; <[256 x i32]*> [#uses=0]
@trace_level = external global i32		; <i32*> [#uses=0]
@.str_9.upgrd.38 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_10.upgrd.39 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@search_failed_high = external global i32		; <i32*> [#uses=0]
@search_failed_low = external global i32		; <i32*> [#uses=0]
@nodes_per_second = external global i32		; <i32*> [#uses=0]
@time_limit = external global i32		; <i32*> [#uses=0]
@easy_move = external global i32		; <i32*> [#uses=0]
@noise_level = external global i32		; <i32*> [#uses=0]
@.str_12.upgrd.40 = external global [34 x i8]		; <[34 x i8]*> [#uses=0]
@.str_136 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@solution_type = external global i32		; <i32*> [#uses=0]
@number_of_solutions = external global i32		; <i32*> [#uses=0]
@solutions = external global [10 x i32]		; <[10 x i32]*> [#uses=0]
@early_exit = external global i32		; <i32*> [#uses=0]
@.str_14.upgrd.41 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_15.upgrd.42 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_16.upgrd.43 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@whisper_value = external global i32		; <i32*> [#uses=0]
@.str_17.upgrd.44 = external global [29 x i8]		; <[29 x i8]*> [#uses=0]
@.str_19.upgrd.45 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@last_mate_score = external global i32		; <i32*> [#uses=0]
@search_depth = external global i32		; <i32*> [#uses=0]
@elapsed_end = external global i32		; <i32*> [#uses=0]
@.str_20.upgrd.46 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_21.upgrd.47 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_22 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str_23.upgrd.48 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_24.upgrd.49 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_25.upgrd.50 = external global [67 x i8]		; <[67 x i8]*> [#uses=0]
@.str_26.upgrd.51 = external global [69 x i8]		; <[69 x i8]*> [#uses=0]
@hash_move = external global [65 x i32]		; <[65 x i32]*> [#uses=0]
@version = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@mode = external global i32		; <i32*> [#uses=0]
@batch_mode = external global i32		; <i32*> [#uses=0]
@crafty_rating = external global i32		; <i32*> [#uses=0]
@opponent_rating = external global i32		; <i32*> [#uses=0]
@pgn_event = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@pgn_site = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@pgn_date = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@pgn_round = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@pgn_white = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@pgn_white_elo = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@pgn_black = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@pgn_black_elo = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@pgn_result = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@number_auto_kibitzers = external global i32		; <i32*> [#uses=0]
@auto_kibitz_list = external global [100 x [20 x i8]]		; <[100 x [20 x i8]]*> [#uses=0]
@number_of_computers = external global i32		; <i32*> [#uses=0]
@computer_list = external global [100 x [20 x i8]]		; <[100 x [20 x i8]]*> [#uses=0]
@number_of_GMs = external global i32		; <i32*> [#uses=0]
@GM_list = external global [100 x [20 x i8]]		; <[100 x [20 x i8]]*> [#uses=0]
@number_of_IMs = external global i32		; <i32*> [#uses=0]
@IM_list = external global [100 x [20 x i8]]		; <[100 x [20 x i8]]*> [#uses=0]
@ics = external global i32		; <i32*> [#uses=0]
@output_format = external global i32		; <i32*> [#uses=0]
@EGTBlimit = external global i32		; <i32*> [#uses=0]
@whisper = external global i32		; <i32*> [#uses=0]
@channel = external global i32		; <i32*> [#uses=0]
@new_game = external global i32		; <i32*> [#uses=0]
@channel_title = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@initialized = external global i32		; <i32*> [#uses=0]
@kibitz = external global i32		; <i32*> [#uses=0]
@post = external global i32		; <i32*> [#uses=0]
@log_id = external global i32		; <i32*> [#uses=0]
@crafty_is_white = external global i32		; <i32*> [#uses=0]
@last_opponent_move = external global i32		; <i32*> [#uses=0]
@search_move = external global i32		; <i32*> [#uses=0]
@time_used = external global i32		; <i32*> [#uses=0]
@time_used_opponent = external global i32		; <i32*> [#uses=0]
@auto_kibitzing = external global i32		; <i32*> [#uses=0]
@test_mode = external global i32		; <i32*> [#uses=0]
@resign = external global i8		; <i8*> [#uses=0]
@resign_counter = external global i8		; <i8*> [#uses=0]
@resign_count = external global i8		; <i8*> [#uses=0]
@draw_counter = external global i8		; <i8*> [#uses=0]
@draw_count = external global i8		; <i8*> [#uses=0]
@tc_moves = external global i32		; <i32*> [#uses=0]
@tc_time = external global i32		; <i32*> [#uses=0]
@tc_time_remaining = external global i32		; <i32*> [#uses=0]
@tc_moves_remaining = external global i32		; <i32*> [#uses=0]
@tc_secondary_moves = external global i32		; <i32*> [#uses=0]
@tc_secondary_time = external global i32		; <i32*> [#uses=0]
@tc_sudden_death = external global i32		; <i32*> [#uses=0]
@tc_operator_time = external global i32		; <i32*> [#uses=0]
@tc_safety_margin = external global i32		; <i32*> [#uses=0]
@force = external global i32		; <i32*> [#uses=0]
@over = external global i32		; <i32*> [#uses=0]
@usage_level = external global i32		; <i32*> [#uses=0]
@audible_alarm = external global i8		; <i8*> [#uses=0]
@ansi = external global i32		; <i32*> [#uses=0]
@book_accept_mask = external global i32		; <i32*> [#uses=0]
@book_reject_mask = external global i32		; <i32*> [#uses=0]
@book_random = external global i32		; <i32*> [#uses=0]
@book_search_trigger = external global i32		; <i32*> [#uses=0]
@learning = external global i32		; <i32*> [#uses=0]
@show_book = external global i32		; <i32*> [#uses=0]
@book_selection_width = external global i32		; <i32*> [#uses=0]
@ponder = external global i32		; <i32*> [#uses=0]
@verbosity_level = external global i32		; <i32*> [#uses=0]
@push_extensions = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_28.upgrd.52 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_3.upgrd.53 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@display = external global %typedef.CHESS_POSITION		; <%typedef.CHESS_POSITION*> [#uses=0]
@.str_4.upgrd.54 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@opponent_start_time = external global i32		; <i32*> [#uses=0]
@.str_8.upgrd.55 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_9.upgrd.56 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_18 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_19.upgrd.57 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_2013 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_21.upgrd.58 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str_22.upgrd.59 = external global [29 x i8]		; <[29 x i8]*> [#uses=0]
@.str_23.upgrd.60 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@whisper_text = external global [500 x i8]		; <[500 x i8]*> [#uses=0]
@.str_24.upgrd.61 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_25.upgrd.62 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_26.upgrd.63 = external global [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str_28.upgrd.64 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str_29 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str_30 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_31 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_32.upgrd.65 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_36 = external global [3 x i8]		; <[3 x i8]*> [#uses=1]
@.str_37 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_44 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_45 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_49 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_52 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@previous_search_value = external global i32		; <i32*> [#uses=0]
@.str_64 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@whisper_depth = external global i32		; <i32*> [#uses=0]
@.str_65 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_66 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@total_moves = external global i32		; <i32*> [#uses=0]
@book_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@books_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@book_lrn_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@position_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@position_lrn_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
@log_filename = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@history_filename = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@book_path = external global [128 x i8]		; <[128 x i8]*> [#uses=0]
@log_path = external global [128 x i8]		; <[128 x i8]*> [#uses=0]
@tb_path = external global [128 x i8]		; <[128 x i8]*> [#uses=0]
@cmd_buffer = external global [512 x i8]		; <[512 x i8]*> [#uses=0]
@root_move = external global i32		; <i32*> [#uses=0]
@hint = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@absolute_time_limit = external global i32		; <i32*> [#uses=0]
@search_time_limit = external global i32		; <i32*> [#uses=0]
@in_check = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@extended_reason = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@current_phase = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@sort_value = external global [256 x i32]		; <[256 x i32]*> [#uses=0]
@next_status = external global [65 x %typedef.NEXT_MOVE]		; <[65 x %typedef.NEXT_MOVE]*> [#uses=0]
@save_hash_key = external global [67 x i64]		; <[67 x i64]*> [#uses=0]
@save_pawn_hash_key = external global [67 x i32]		; <[67 x i32]*> [#uses=0]
@pawn_advance = external global [8 x i32]		; <[8 x i32]*> [#uses=0]
@bit_move = external global i64		; <i64*> [#uses=0]
@.str_1.upgrd.66 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_2.upgrd.67 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_3.upgrd.68 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_1.upgrd.69 = external global [34 x i8]		; <[34 x i8]*> [#uses=0]
@.str_2.upgrd.70 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_2.upgrd.71 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_1.upgrd.72 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_2.upgrd.73 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_3.upgrd.74 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_4.upgrd.75 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_5.upgrd.76 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_615 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_7.upgrd.77 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_10.upgrd.78 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_11.upgrd.79 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_12.upgrd.80 = external global [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str_1318 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_1419 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_15.upgrd.81 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_16.upgrd.82 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_19.upgrd.83 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_20.upgrd.84 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_2222 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_2323 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_25.upgrd.85 = external global [29 x i8]		; <[29 x i8]*> [#uses=0]
@.str_27 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_28.upgrd.86 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_29.upgrd.87 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_30.upgrd.88 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_31.upgrd.89 = external global [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str_32.upgrd.90 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_33 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_34 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_35 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_36.upgrd.91 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_37.upgrd.92 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_38 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_41 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_42 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_43 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_44.upgrd.93 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_4525 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_46 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_47 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_48 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_49.upgrd.94 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_50 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_51 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_52.upgrd.95 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_53 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_54 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_55 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_56 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_57 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_58 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_59 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_60 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_61 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_62 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_63 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_64.upgrd.96 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_66.upgrd.97 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_67 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_68 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_69 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_71 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_72 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_73 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_74 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_75 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_81 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_83 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_84 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_86 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_87 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_89 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_90 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_91 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_92 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_94 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_95 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_96 = external global [34 x i8]		; <[34 x i8]*> [#uses=0]
@.str_97 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_98 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_100 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_101 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_102 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_103 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_104 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_105 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_106 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_107 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_108 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_109 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_110 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_111 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_112 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_113 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_114 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_115 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_116 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_117 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_118 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_119 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_120 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_121 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_122 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_123 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_124 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_125 = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@.str_126 = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@.str_127 = external global [69 x i8]		; <[69 x i8]*> [#uses=0]
@.str_128 = external global [66 x i8]		; <[66 x i8]*> [#uses=0]
@.str_129 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_130 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_131 = external global [67 x i8]		; <[67 x i8]*> [#uses=0]
@.str_132 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_133 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_134 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_135 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_136.upgrd.98 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_137 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_138 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_139 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_140 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_141 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_142 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_143 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_144 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_145 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_146 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_147 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_148 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_149 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_150 = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@.str_151 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_152 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_153 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_154 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_156 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_157 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str_158 = external global [71 x i8]		; <[71 x i8]*> [#uses=0]
@.str_159 = external global [72 x i8]		; <[72 x i8]*> [#uses=0]
@.str_160 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_161 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_162 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_163 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_164 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_165 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_166 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_167 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_168 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_169 = external global [65 x i8]		; <[65 x i8]*> [#uses=0]
@.str_170 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_171 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_172 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_173 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_174 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_175 = external global [70 x i8]		; <[70 x i8]*> [#uses=0]
@.str_176 = external global [67 x i8]		; <[67 x i8]*> [#uses=0]
@.str_177 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_178 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_180 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_181 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_182 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_183 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_184 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_185 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_186 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_187 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_188 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_189 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_190 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_191 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_192 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_193 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_194 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_195 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_196 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_197 = external global [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str_198 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_201 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_202 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_203 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_204 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_206 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_207 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_208 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_209 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_210 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_211 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_213 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_214 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_215 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_216 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_218 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_219 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_220 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_221 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_222 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_223 = external global [66 x i8]		; <[66 x i8]*> [#uses=0]
@.str_224 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_225 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_226 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_227 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_228 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_229 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_230 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_231 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_232 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_233 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_234 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_235 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_236 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_237 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_238 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_239 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_240 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_241 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_242 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_243 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_245 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_246 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_247 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_248 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_249 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_250 = external global [45 x i8]		; <[45 x i8]*> [#uses=0]
@.str_253 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_254 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_256 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_258 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_259 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_261 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_262 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_263 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_266 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_267 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_268 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_270 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_271 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_272 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_273 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_274 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_275 = external global [44 x i8]		; <[44 x i8]*> [#uses=0]
@.str_276 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_277 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_278 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_279 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_280 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_281 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_282 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_283 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_284 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_285 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_286 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_287 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_288 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_289 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_290 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_291 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_292 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_293 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_294 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_295 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_296 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_297 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_298 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_299 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_300 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_301 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_302 = external global [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str_304 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_305 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_306 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_308 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_310 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_311 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_312 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_313 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_314 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_315 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_316 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_317 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_319 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_320 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_321 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_322 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_323 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_325 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_327 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_328 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_329 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_330 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_331 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_332 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_333 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_334 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_335 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_336 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_337 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_338 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_339 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_340 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_341 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_342 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_343 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_344 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_345 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_346 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_347 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_348 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_349 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_350 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_351 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_352 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_353 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_354 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_355 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_356 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_357 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_358 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_359 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_360 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_362 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_363 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_364 = external global [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str_365 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_366 = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@.str_367 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_368 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_369 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_370 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_371 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_372 = external global [58 x i8]		; <[58 x i8]*> [#uses=0]
@.str_373 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_374 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_375 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_376 = external global [63 x i8]		; <[63 x i8]*> [#uses=0]
@.str_377 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_378 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_379 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_380 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_381 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_382 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_383 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_384 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_385 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_387 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_388 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_389 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_390 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_391 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_392 = external global [71 x i8]		; <[71 x i8]*> [#uses=0]
@.str_393 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_394 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_395 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_396 = external global [60 x i8]		; <[60 x i8]*> [#uses=0]
@.str_397 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_398 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_399 = external global [67 x i8]		; <[67 x i8]*> [#uses=0]
@.str_400 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_401 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_402 = external global [62 x i8]		; <[62 x i8]*> [#uses=0]
@.str_403 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_404 = external global [59 x i8]		; <[59 x i8]*> [#uses=0]
@.str_405 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_406 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_407 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_408 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_409 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_410 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_411 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_412 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_413 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_414 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_421 = external global [53 x i8]		; <[53 x i8]*> [#uses=0]
@.str_422 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_423 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_424 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_426 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_427 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_429 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_430 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_431 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_432 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_433 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_434 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_435 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_436 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_437 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_438 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_440 = external global [44 x i8]		; <[44 x i8]*> [#uses=0]
@.str_445 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_446 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_447 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_448 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_449 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_450 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_451 = external global [44 x i8]		; <[44 x i8]*> [#uses=0]
@.str_452 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_453 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_454 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_455 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_456 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_459 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_460 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_461 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_462 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_463 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_466 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_467 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_468 = external global [45 x i8]		; <[45 x i8]*> [#uses=0]
@.str_469 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_470 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str_474 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_477 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_480 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_483 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_485 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_487 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_490 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_494 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_495 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_497 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_498 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str_507 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_508 = external global [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str_509 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_510 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_511 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_512 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_513 = external global [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str_514 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_515 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_516 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_517 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_519 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_520 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_521 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_522 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_523 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_524 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_525 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_526 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_527 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_528 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_529 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_530 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_531 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_532 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_533 = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str_534 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_535 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_536 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_537 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_539 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_540 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_541 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_542 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_543 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_544 = external global [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str_546 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_550 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_551 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_552 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str_553 = external global [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str_554 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_555 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_556 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_557 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_559 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_560 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_562 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_564 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_565 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_567 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_568 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_570 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_571 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_572 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_574 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_576 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_577 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_578 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_579 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_580 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_581 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_582 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_583 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_584 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_586 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_587 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_589 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_590 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_591 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_592 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_596 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_597 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_598 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_599 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_605 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_610 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_613 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_616 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_621 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_622 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_623 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_624 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_625 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_626 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_628 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_629 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_630 = external global [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str_631 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_632 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_633 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_634 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_635 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_636 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_637 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_639 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_643 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_644 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_645 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_646 = external global [1 x i8]		; <[1 x i8]*> [#uses=0]
@.str_649 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_654 = external global [2 x i8]		; <[2 x i8]*> [#uses=1]
@.str_656 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_658 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_660 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_662 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_664 = external global [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str_666 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_667 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_669 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_670 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_671 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_672 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_674 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_675 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_676 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_680 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_682 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_683 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_684 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_685 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_686 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_687 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_688 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_689 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_690 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_691 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_692 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_694 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_695 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_697 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_698 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_700 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_701 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_702 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_703 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_704 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_707 = external global [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str_708 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_709 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_710 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_711 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_722 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_723 = external global [34 x i8]		; <[34 x i8]*> [#uses=0]
@.str_726 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_727 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_728 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_729 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_730 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_732 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_734 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_735 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_736 = external global [61 x i8]		; <[61 x i8]*> [#uses=0]
@.str_738 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_739 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_740 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_741 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_742 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_743 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_744 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str_745 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_747 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_748 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_750 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@"\01text_move.0__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@"\01new_text.1__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_1.upgrd.99 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@"\01text_move.2__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_1.upgrd.100 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_2.upgrd.101 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_3.upgrd.102 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_130.upgrd.103 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_231.upgrd.104 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_3.upgrd.105 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_4.upgrd.106 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_7.upgrd.107 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@"\01hashing_pawns.0__" = external global i32		; <i32*> [#uses=0]
@"\01hashing_opening.1__" = external global i32		; <i32*> [#uses=0]
@"\01hashing_middle_game.2__" = external global i32		; <i32*> [#uses=0]
@"\01hashing_end_game.3__" = external global i32		; <i32*> [#uses=0]
@"\01last_wtm.4__" = external global i32		; <i32*> [#uses=0]
@.str_1.upgrd.108 = external global [37 x i8]		; <[37 x i8]*> [#uses=0]
@.str_1.upgrd.109 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_1.upgrd.110 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_2.upgrd.111 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_3.upgrd.112 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_4.upgrd.113 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_5.upgrd.114 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_6.upgrd.115 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_7.upgrd.116 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_934 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_1.upgrd.117 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_3.upgrd.118 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_1.upgrd.119 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_2.upgrd.120 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_4.upgrd.121 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_5.upgrd.122 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_1.upgrd.123 = external global [11 x i8]		; <[11 x i8]*> [#uses=0]
@.str_2.upgrd.124 = external global [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str_7.upgrd.125 = external global [29 x i8]		; <[29 x i8]*> [#uses=0]
@.str_10.upgrd.126 = external global [34 x i8]		; <[34 x i8]*> [#uses=0]
@.str_1141 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_12.upgrd.127 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_14.upgrd.128 = external global [20 x i8]		; <[20 x i8]*> [#uses=0]
@.str_1542 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.ctor_1.upgrd.129 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_1.upgrd.130 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_3.upgrd.131 = external global [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str_4.upgrd.132 = external global [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str_5.upgrd.133 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_6.upgrd.134 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str_143.upgrd.135 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_2.upgrd.136 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_1.upgrd.137 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_2.upgrd.138 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@bit_move44 = external global i64		; <i64*> [#uses=0]
@.str_1.upgrd.139 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_248.upgrd.140 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_349.upgrd.141 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.ctor_1.upgrd.142 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_5.upgrd.143 = external global [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str_6.upgrd.144 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_751 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_852 = external global [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str_9.upgrd.145 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_10.upgrd.146 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@"\01out.0__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_1153 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_12.upgrd.147 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_13.upgrd.148 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_14.upgrd.149 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_15.upgrd.150 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_16.upgrd.151 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_17.upgrd.152 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@"\01out.1__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_18.upgrd.153 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_19.upgrd.154 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_20.upgrd.155 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_21.upgrd.156 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_2254 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_2355 = external global [8 x i8]		; <[8 x i8]*> [#uses=0]
@.str_24.upgrd.157 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str_25.upgrd.158 = external global [45 x i8]		; <[45 x i8]*> [#uses=0]
@.str_26.upgrd.159 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@"\01out.2__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_31.upgrd.160 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@"\01out.3__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@"\01out.4__" = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_3457 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_35.upgrd.161 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_36.upgrd.162 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_37.upgrd.163 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_41.upgrd.164 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_45.upgrd.165 = external global [55 x i8]		; <[55 x i8]*> [#uses=0]
@"\01save_book_selection_width.5__" = external global i32		; <i32*> [#uses=0]
@"\01save_book_random.6__" = external global i32		; <i32*> [#uses=0]
@"\01save_whisper.7__" = external global i32		; <i32*> [#uses=0]
@"\01save_kibitz.8__" = external global i32		; <i32*> [#uses=0]
@"\01save_channel.9__" = external global i32		; <i32*> [#uses=0]
@"\01save_resign.10" = external global i32		; <i32*> [#uses=0]
@"\01save_resign_count.11" = external global i32		; <i32*> [#uses=0]
@"\01save_draw_count.12" = external global i32		; <i32*> [#uses=0]
@"\01save_learning.13" = external global i32		; <i32*> [#uses=0]
@.str_49.upgrd.166 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_51.upgrd.167 = external global [44 x i8]		; <[44 x i8]*> [#uses=0]
@"\01x.14" = external global [55 x i32]		; <[55 x i32]*> [#uses=0]
@"\01init.15.b" = external global i1		; <i1*> [#uses=0]
@"\01y.16" = external global [55 x i32]		; <[55 x i32]*> [#uses=0]
@"\01j.17" = external global i32		; <i32*> [#uses=0]
@"\01k.18" = external global i32		; <i32*> [#uses=0]
@.str_52.upgrd.168 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@"\01text.19" = external global [128 x i8]		; <[128 x i8]*> [#uses=0]
@.str_5659 = external global [12 x i8]		; <[12 x i8]*> [#uses=0]
@.str_62.upgrd.169 = external global [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str_6662 = external global [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str_68.upgrd.170 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_69.upgrd.171 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_70 = external global [16 x i8]		; <[16 x i8]*> [#uses=0]
@.str_72.upgrd.172 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_74.upgrd.173 = external global [23 x i8]		; <[23 x i8]*> [#uses=0]
@.str_76 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_78 = external global [57 x i8]		; <[57 x i8]*> [#uses=0]
@.str_80 = external global [45 x i8]		; <[45 x i8]*> [#uses=0]
@.str_82 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_84.upgrd.174 = external global [10 x i8]		; <[10 x i8]*> [#uses=0]
@.str_86.upgrd.175 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_88 = external global [7 x i8]		; <[7 x i8]*> [#uses=0]
@.str_90.upgrd.176 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str_92.upgrd.177 = external global [19 x i8]		; <[19 x i8]*> [#uses=0]
@.str_94.upgrd.178 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_95.upgrd.179 = external global [48 x i8]		; <[48 x i8]*> [#uses=0]
@.str_97.upgrd.180 = external global [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str_98.upgrd.181 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_100.upgrd.182 = external global [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str_163.upgrd.183 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_2.upgrd.184 = external global [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str_3.upgrd.185 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_4.upgrd.186 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_5.upgrd.187 = external global [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str_6.upgrd.188 = external global [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str_7.upgrd.189 = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@.str_8.upgrd.190 = external global [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str_9.upgrd.191 = external global [54 x i8]		; <[54 x i8]*> [#uses=0]
@.str_10.upgrd.192 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_11.upgrd.193 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_12.upgrd.194 = external global [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str_13.upgrd.195 = external global [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str_14.upgrd.196 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_15.upgrd.197 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_16.upgrd.198 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_17.upgrd.199 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_18.upgrd.200 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_19.upgrd.201 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str_20.upgrd.202 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_22.upgrd.203 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_23.upgrd.204 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_24.upgrd.205 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_26.upgrd.206 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_27.upgrd.207 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_28.upgrd.208 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_30.upgrd.209 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_31.upgrd.210 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_32.upgrd.211 = external global [36 x i8]		; <[36 x i8]*> [#uses=0]
@.str_33.upgrd.212 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_34.upgrd.213 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_3565 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_36.upgrd.214 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_37.upgrd.215 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str_38.upgrd.216 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str_39 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_40 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_41.upgrd.217 = external global [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str_42.upgrd.218 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_43.upgrd.219 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str_44.upgrd.220 = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str_45.upgrd.221 = external global [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str_46.upgrd.222 = external global [35 x i8]		; <[35 x i8]*> [#uses=0]
@.str_47.upgrd.223 = external global [50 x i8]		; <[50 x i8]*> [#uses=0]
@.str_48.upgrd.224 = external global [26 x i8]		; <[26 x i8]*> [#uses=0]
@.str_49.upgrd.225 = external global [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str_50.upgrd.226 = external global [15 x i8]		; <[15 x i8]*> [#uses=0]
@.str_51.upgrd.227 = external global [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str_52.upgrd.228 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_53.upgrd.229 = external global [9 x i8]		; <[9 x i8]*> [#uses=0]

declare i64 @AttacksFrom(i32, i32)

declare i64 @AttacksTo(i32)

declare i32 @Attacked(i32, i32)

declare i64 @Mask(i32)

declare i32 @PopCnt(i64)

declare i32 @FirstOne(i64)

declare i32 @LastOne(i64)

declare i32 @DrawScore()

declare i32 @Drawn(i32)

declare i8* @strchr(i8*, i32)

declare i32 @strcmp(i8*, i8*)

declare i32 @strlen(i8*)

declare i32 @printf(i8*, ...)

declare void @Edit()

declare void @llvm.memcpy(i8*, i8*, i32, i32)

declare i32 @fflush(%struct.__sFILE*)

declare i32 @Read(i32, i8*)

declare i32 @ReadParse(i8*, i8**, i8*)

declare void @DisplayChessBoard(%struct.__sFILE*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)

declare void @SetChessBitBoards(%typedef.SEARCH_POSITION*)

declare i32 @EnPrise(i32, i32)

declare i64 @SwapXray(i64, i32, i32)

declare i32 @Evaluate(i32, i32, i32, i32)

declare i32 @EvaluateMate()

declare i32 @EvaluatePawns()

declare i32 @EvaluatePassedPawns()

declare i32 @EvaluatePassedPawnRaces(i32)

declare i32 @Swap(i32, i32, i32)

declare i32 @EvaluateDevelopment(i32)

declare i32 @EvaluateDraws()

declare i32 @HasOpposition(i32, i32, i32)

declare void @HistoryBest(i32, i32, i32)

declare void @HistoryRefutation(i32, i32, i32)

declare i32 @sprintf(i8*, i8*, ...)

declare void @Initialize(i32)

declare void @InitializeZeroMasks()

declare void @InitializeMasks()

declare void @InitializeRandomHash()

declare void @InitializeAttackBoards()

declare void @InitializePawnMasks()

declare void @InitializePieceMasks()

declare void @InitializeChessBoard(%typedef.SEARCH_POSITION*)

declare %struct.__sFILE* @fopen(i8*, i8*)

define i32 @Option() {
no_exit.53.outer:
	%tmp.4747 = shl i32 7, 3		; <i32> [#uses=1]
	%tmp.4779 = icmp eq %struct.__sFILE* getelementptr ([0 x %struct.__sFILE]* @__sF, i32 0, i32 1), null		; <i1> [#uses=2]
	br label %no_exit.53
no_exit.53:		; preds = %else.166, %else.168, %then.360, %no_exit.53.outer
	%file.2.3.3.ph = phi i32 [ 0, %no_exit.53.outer ], [ %inc.551688, %then.360 ], [ %inc.551701, %else.168 ], [ %file.2.3.3.ph, %else.166 ]		; <i32> [#uses=2]
	%nempty.5.3.ph = phi i32 [ 0, %no_exit.53.outer ], [ %nempty.5.3, %then.360 ], [ %nempty.5.3, %else.168 ], [ %nempty.5.3.ph, %else.166 ]		; <i32> [#uses=2]
	%indvar2053.ui = phi i32 [ 0, %no_exit.53.outer ], [ 0, %then.360 ], [ 0, %else.168 ], [ %indvar.next2054, %else.166 ]		; <i32> [#uses=2]
	%indvar2053 = bitcast i32 %indvar2053.ui to i32		; <i32> [#uses=2]
	%file.2.3.3 = add i32 %indvar2053, %file.2.3.3.ph		; <i32> [#uses=4]
	%nempty.5.3 = add i32 %indvar2053, %nempty.5.3.ph		; <i32> [#uses=3]
	%tmp.4749 = add i32 %file.2.3.3, %tmp.4747		; <i32> [#uses=1]
	%tmp.4750 = getelementptr %typedef.CHESS_POSITION* @search, i32 0, i32 22, i32 %tmp.4749		; <i8*> [#uses=3]
	%tmp.4751 = load i8* %tmp.4750		; <i8> [#uses=1]
	%tmp.4752 = icmp eq i8 %tmp.4751, 0		; <i1> [#uses=1]
	br i1 %tmp.4752, label %else.166, label %then.357
then.357:		; preds = %no_exit.53
	%tmp.4755 = icmp eq i32 %nempty.5.3, 0		; <i1> [#uses=1]
	br i1 %tmp.4755, label %endif.358, label %then.358
then.358:		; preds = %then.357
	ret i32 0
endif.358:		; preds = %then.357
	br i1 %tmp.4779, label %else.168, label %then.360
then.360:		; preds = %endif.358
	%tmp.4791 = load i8* %tmp.4750		; <i8> [#uses=1]
	%tmp.4792 = sext i8 %tmp.4791 to i32		; <i32> [#uses=1]
	%tmp.4793 = add i32 %tmp.4792, 7		; <i32> [#uses=1]
	%tmp.4794 = getelementptr [15 x i8]* null, i32 0, i32 %tmp.4793		; <i8*> [#uses=1]
	%tmp.4795 = load i8* %tmp.4794		; <i8> [#uses=1]
	%tmp.4796 = sext i8 %tmp.4795 to i32		; <i32> [#uses=1]
	%tmp.4781 = call i32 (%struct.__sFILE*, i8*, ...)* @fprintf( %struct.__sFILE* getelementptr ([0 x %struct.__sFILE]* @__sF, i32 0, i32 1), i8* getelementptr ([3 x i8]* @.str_36, i32 0, i32 0), i32 %tmp.4796 )		; <i32> [#uses=0]
	%inc.551688 = add i32 %file.2.3.3, 1		; <i32> [#uses=2]
	%tmp.47421699 = icmp slt i32 %inc.551688, 8		; <i1> [#uses=1]
	br i1 %tmp.47421699, label %no_exit.53, label %loopexit.56
else.168:		; preds = %endif.358
	%tmp.4799 = call i32 @strlen( i8* getelementptr ([80 x i8]* @initial_position, i32 0, i32 0) )		; <i32> [#uses=2]
	%gep.upgrd.230 = zext i32 %tmp.4799 to i64		; <i64> [#uses=1]
	%tmp.4802 = getelementptr [80 x i8]* @initial_position, i32 0, i64 %gep.upgrd.230		; <i8*> [#uses=1]
	%tmp.4811 = load i8* %tmp.4750		; <i8> [#uses=1]
	%tmp.4812 = sext i8 %tmp.4811 to i32		; <i32> [#uses=1]
	%tmp.4813 = add i32 %tmp.4812, 7		; <i32> [#uses=1]
	%tmp.4814 = getelementptr [15 x i8]* null, i32 0, i32 %tmp.4813		; <i8*> [#uses=1]
	%tmp.4815 = load i8* %tmp.4814		; <i8> [#uses=1]
	store i8 %tmp.4815, i8* %tmp.4802
	%tmp.4802.sum = add i32 %tmp.4799, 1		; <i32> [#uses=1]
	%gep.upgrd.231 = zext i32 %tmp.4802.sum to i64		; <i64> [#uses=1]
	%tmp.4802.end = getelementptr [80 x i8]* @initial_position, i32 0, i64 %gep.upgrd.231		; <i8*> [#uses=1]
	store i8 0, i8* %tmp.4802.end
	%inc.551701 = add i32 %file.2.3.3, 1		; <i32> [#uses=2]
	%tmp.47421703 = icmp slt i32 %inc.551701, 8		; <i1> [#uses=1]
	br i1 %tmp.47421703, label %no_exit.53, label %loopexit.56
else.166:		; preds = %no_exit.53
	%inc.55 = add i32 %file.2.3.3, 1		; <i32> [#uses=1]
	%tmp.47421705 = icmp slt i32 %inc.55, 8		; <i1> [#uses=1]
	%indvar.next2054 = add i32 %indvar2053.ui, 1		; <i32> [#uses=1]
	br i1 %tmp.47421705, label %no_exit.53, label %loopexit.56
loopexit.56:		; preds = %else.166, %else.168, %then.360
	br i1 %tmp.4779, label %else.169, label %then.361
then.361:		; preds = %loopexit.56
	%tmp.4822 = call i32 @fwrite( i8* getelementptr ([2 x i8]* @.str_654, i32 0, i32 0), i32 1, i32 1, %struct.__sFILE* getelementptr ([0 x %struct.__sFILE]* @__sF, i32 0, i32 1) )		; <i32> [#uses=0]
	%dec.101707 = add i32 7, -1		; <i32> [#uses=1]
	%tmp.47391709 = icmp sgt i32 %dec.101707, -1		; <i1> [#uses=0]
	ret i32 0
else.169:		; preds = %loopexit.56
	%tmp.4827 = call i32 @strlen( i8* getelementptr ([80 x i8]* @initial_position, i32 0, i32 0) )		; <i32> [#uses=2]
	%gep.upgrd.232 = zext i32 %tmp.4827 to i64		; <i64> [#uses=1]
	%tmp.4830 = getelementptr [80 x i8]* @initial_position, i32 0, i64 %gep.upgrd.232		; <i8*> [#uses=1]
	store i8 47, i8* %tmp.4830
	%tmp.4830.sum = add i32 %tmp.4827, 1		; <i32> [#uses=1]
	%gep.upgrd.233 = zext i32 %tmp.4830.sum to i64		; <i64> [#uses=1]
	%tmp.4830.end = getelementptr [80 x i8]* @initial_position, i32 0, i64 %gep.upgrd.233		; <i8*> [#uses=1]
	store i8 0, i8* %tmp.4830.end
	%dec.10 = add i32 7, -1		; <i32> [#uses=1]
	%tmp.47391711 = icmp sgt i32 %dec.10, -1		; <i1> [#uses=0]
	ret i32 0
}

declare void @InitializeHashTables()

declare i32 @InitializeFindAttacks(i32, i32, i32)

declare void @SetBoard(i32, i8**, i32)

declare i32 @KingPawnSquare(i32, i32, i32, i32)

declare i64 @Random64()

declare i32 @Random32()

declare i8* @strcpy(i8*, i8*)

declare i32 @InputMove(i8*, i32, i32, i32, i32)

declare i32 @InputMoveICS(i8*, i32, i32, i32, i32)

declare i32* @GenerateCaptures(i32, i32, i32*)

declare i32* @GenerateNonCaptures(i32, i32, i32*)

declare void @MakeMove(i32, i32, i32)

declare void @UnMakeMove(i32, i32, i32)

declare void @Interrupt(i32)

declare i32 @GetTime(i32)

declare i8* @DisplayTime(i32)

declare i8* @OutputMoveICS(i32*)

declare void @Delay(i32)

declare i32 @fprintf(%struct.__sFILE*, i8*, ...)

declare void @SignalInterrupt(i32)

declare void (i32)* @signal(i32, void (i32)*)

declare i32 @Iterate(i32, i32, i32)

declare void @PreEvaluate(i32)

declare void @RootMoveList(i32)

declare i8* @OutputMove(i32*, i32, i32)

declare void @TimeSet(i32)

declare void @StorePV(i32, i32)

declare i32 @SearchRoot(i32, i32, i32, i32)

declare void @Whisper(i32, i32, i32, i32, i32, i32, i8*)

declare i8* @DisplayEvaluation(i32)

declare i32 @LookUp(i32, i32, i32, i32*, i32*)

declare i8* @strstr(i8*, i8*)

declare i32 @main(i32, i8**)

declare void @__main()

declare i32 @atoi(i8*)

declare void @NewGame(i32)

declare i32 @Ponder(i32)

declare i32 @fseek(%struct.__sFILE*, i32, i32)

declare void @MakeMoveRoot(i32, i32)

declare i32 @RepetitionDraw(i32)

declare i8* @Reverse()

declare i8* @Normal()

declare void @TimeAdjust(i32, i32)

declare void @ValidatePosition(i32, i32, i8*)

declare i32 @ValidMove(i32, i32, i32)

declare i32* @GenerateCheckEvasions(i32, i32, i32*)

declare i64 @InterposeSquares(i32, i32, i32)

declare i32 @PinnedOnKing(i32, i32)

declare i32 @NextMove(i32, i32)

declare i32 @NextEvasion(i32, i32)

declare i32 @NextRootMove(i32)

declare i32 @TimeCheck(i32)

declare i32 @strncmp(i8*, i8*, i32)

declare void @exit(i32)

declare i32 @OptionMatch(i8*, i8*)

declare i32 @fclose(%struct.__sFILE*)

declare i32 @ParseTime(i8*)

declare i8* @DisplayHHMM(i32)

declare void @DisplayPieceBoards(i32*, i32*)

declare i32 @fscanf(%struct.__sFILE*, i8*, ...)

declare i32 @feof(%struct.__sFILE*)

declare i8* @fgets(i8*, i32, %struct.__sFILE*)

declare i32 @remove(i8*)

declare i32 @__tolower(i32)

declare i32 @clock()

declare void @OptionPerft(i32, i32, i32)

declare void @Phase()

declare i32 @ReadNextMove(i8*, i32, i32)

declare i32 @time(i32*)

declare %struct.tm* @localtime(i32*)

declare i8* @gets(i8*)

declare i32 @OutputGood(i8*, i32, i32)

declare i32 @CheckInput()

declare void @ClearHashTables()

declare i32 @Quiesce(i32, i32, i32, i32)

declare void @SearchTrace(i32, i32, i32, i32, i32, i8*, i32)

declare i32 @RepetitionCheck(i32, i32)

declare void @ResignOrDraw(i32, i32)

declare i32 @Search(i32, i32, i32, i32, i32, i32)

declare void @StoreRefutation(i32, i32, i32, i32)

declare void @StoreBest(i32, i32, i32, i32, i32)

declare void @SearchOutput(i32, i32)

declare i32 @strspn(i8*, i8*)

declare i32 @isatty(i32)

declare i32 @fileno(%struct.__sFILE*)

declare void @llvm.memset(i8*, i8, i32, i32)

declare i32 @select(i32, %struct.fd_set*, %struct.fd_set*, %struct.fd_set*, %struct.timeval*)

declare void @DisplayBitBoard(i64)

declare i8* @DisplayEvaluationWhisper(i32)

declare i8* @DisplayTimeWhisper(i32)

declare void @Display64bitWord(i64)

declare void @Display2BitBoards(i64, i64)

declare void @DisplayChessMove(i8*, i32)

declare void @llvm.memmove(i8*, i8*, i32, i32)

declare void @ReadClear()

declare i8* @strtok(i8*, i8*)

declare i32 @SpecReadRaw()

declare i32 @read(i32, i8*, i32)

declare i32* @__error()

declare i32 @ReadChessMove(%struct.__sFILE*, i32, i32)

declare i64 @ValidateComputeBishopAttacks(i32)

declare i64 @ValidateComputeRookAttacks(i32)

declare i8* @memchr(i8*, i32, i32)

declare i32 @fwrite(i8*, i32, i32, %struct.__sFILE*)
