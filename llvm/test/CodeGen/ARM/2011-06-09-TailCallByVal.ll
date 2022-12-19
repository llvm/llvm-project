; RUN: llc < %s -relocation-model=pic -mcpu=cortex-a8 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

%struct._RuneCharClass = type { [14 x i8], i32 }
%struct._RuneEntry = type { i32, i32, i32, ptr }
%struct._RuneLocale = type { [8 x i8], [32 x i8], ptr, ptr, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, ptr, i32, i32, ptr }
%struct._RuneRange = type { i32, ptr }
%struct.__collate_st_chain_pri = type { [10 x i32], [2 x i32] }
%struct.__collate_st_char_pri = type { [2 x i32] }
%struct.__collate_st_info = type { [2 x i8], i8, i8, [2 x i32], [2 x i32], i32, i32 }
%struct.__collate_st_large_char_pri = type { i32, %struct.__collate_st_char_pri }
%struct.__collate_st_subst = type { i32, [10 x i32] }
%struct.__xlocale_st_collate = type { i32, ptr, [32 x i8], %struct.__collate_st_info, [2 x ptr], ptr, ptr, [256 x %struct.__collate_st_char_pri] }
%struct.__xlocale_st_messages = type { i32, ptr, ptr, %struct.lc_messages_T }
%struct.__xlocale_st_monetary = type { i32, ptr, ptr, %struct.lc_monetary_T }
%struct.__xlocale_st_numeric = type { i32, ptr, ptr, %struct.lc_numeric_T }
%struct.__xlocale_st_runelocale = type { i32, ptr, [32 x i8], i32, i32, ptr, ptr, ptr, ptr, ptr, i32, %struct._RuneLocale }
%struct.__xlocale_st_time = type { i32, ptr, ptr, %struct.lc_time_T }
%struct._xlocale = type { i32, ptr, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, i32, i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.lconv }
%struct.lc_messages_T = type { ptr, ptr, ptr, ptr }
%struct.lc_monetary_T = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.lc_numeric_T = type { ptr, ptr, ptr }
%struct.lc_time_T = type { [12 x ptr], [12 x ptr], [7 x ptr], [7 x ptr], ptr, ptr, ptr, ptr, ptr, ptr, [12 x ptr], ptr, ptr }
%struct.lconv = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%union.__mbstate_t = type { i64, [120 x i8] }

@"\01_fnmatch.initial" = external constant %union.__mbstate_t, align 4

; CHECK: _fnmatch
; CHECK: bl _fnmatch1

define i32 @"\01_fnmatch"(ptr %pattern, ptr %string, i32 %flags) nounwind optsize {
entry:
  %call4 = tail call i32 @fnmatch1(ptr %pattern, ptr %string, ptr %string, i32 %flags, ptr byval(%union.__mbstate_t) @"\01_fnmatch.initial", ptr byval(%union.__mbstate_t) @"\01_fnmatch.initial", ptr undef, i32 64) optsize
  ret i32 %call4
}

declare i32 @fnmatch1(ptr, ptr, ptr, i32, ptr byval(%union.__mbstate_t), ptr byval(%union.__mbstate_t), ptr, i32) nounwind optsize
