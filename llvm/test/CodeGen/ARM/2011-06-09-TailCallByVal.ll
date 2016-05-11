; RUN: llc < %s -relocation-model=pic -mcpu=cortex-a8 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

%struct._RuneCharClass = type { [14 x i8], i32 }
%struct._RuneEntry = type { i32, i32, i32, i32* }
%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i32, i8**)*, i32 (i32, i8*, i32, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32, i32, %struct._RuneCharClass* }
%struct._RuneRange = type { i32, %struct._RuneEntry* }
%struct.__collate_st_chain_pri = type { [10 x i32], [2 x i32] }
%struct.__collate_st_char_pri = type { [2 x i32] }
%struct.__collate_st_info = type { [2 x i8], i8, i8, [2 x i32], [2 x i32], i32, i32 }
%struct.__collate_st_large_char_pri = type { i32, %struct.__collate_st_char_pri }
%struct.__collate_st_subst = type { i32, [10 x i32] }
%struct.__xlocale_st_collate = type { i32, void (i8*)*, [32 x i8], %struct.__collate_st_info, [2 x %struct.__collate_st_subst*], %struct.__collate_st_chain_pri*, %struct.__collate_st_large_char_pri*, [256 x %struct.__collate_st_char_pri] }
%struct.__xlocale_st_messages = type { i32, void (i8*)*, i8*, %struct.lc_messages_T }
%struct.__xlocale_st_monetary = type { i32, void (i8*)*, i8*, %struct.lc_monetary_T }
%struct.__xlocale_st_numeric = type { i32, void (i8*)*, i8*, %struct.lc_numeric_T }
%struct.__xlocale_st_runelocale = type { i32, void (i8*)*, [32 x i8], i32, i32, i32 (i32*, i8*, i32, %union.__mbstate_t*, %struct._xlocale*)*, i32 (%union.__mbstate_t*, %struct._xlocale*)*, i32 (i32*, i8**, i32, i32, %union.__mbstate_t*, %struct._xlocale*)*, i32 (i8*, i32, %union.__mbstate_t*, %struct._xlocale*)*, i32 (i8*, i32**, i32, i32, %union.__mbstate_t*, %struct._xlocale*)*, i32, %struct._RuneLocale }
%struct.__xlocale_st_time = type { i32, void (i8*)*, i8*, %struct.lc_time_T }
%struct._xlocale = type { i32, void (i8*)*, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, %union.__mbstate_t, i32, i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, %struct.__xlocale_st_collate*, %struct.__xlocale_st_runelocale*, %struct.__xlocale_st_messages*, %struct.__xlocale_st_monetary*, %struct.__xlocale_st_numeric*, %struct._xlocale*, %struct.__xlocale_st_time*, %struct.lconv }
%struct.lc_messages_T = type { i8*, i8*, i8*, i8* }
%struct.lc_monetary_T = type { i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }
%struct.lc_numeric_T = type { i8*, i8*, i8* }
%struct.lc_time_T = type { [12 x i8*], [12 x i8*], [7 x i8*], [7 x i8*], i8*, i8*, i8*, i8*, i8*, i8*, [12 x i8*], i8*, i8* }
%struct.lconv = type { i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%union.__mbstate_t = type { i64, [120 x i8] }

@"\01_fnmatch.initial" = external constant %union.__mbstate_t, align 4

; CHECK: _fnmatch
; CHECK: bl _fnmatch1

define i32 @"\01_fnmatch"(i8* %pattern, i8* %string, i32 %flags) nounwind optsize {
entry:
  %call4 = tail call i32 @fnmatch1(i8* %pattern, i8* %string, i8* %string, i32 %flags, %union.__mbstate_t* byval @"\01_fnmatch.initial", %union.__mbstate_t* byval @"\01_fnmatch.initial", %struct._xlocale* undef, i32 64) optsize
  ret i32 %call4
}

declare i32 @fnmatch1(i8*, i8*, i8*, i32, %union.__mbstate_t* byval, %union.__mbstate_t* byval, %struct._xlocale*, i32) nounwind optsize
