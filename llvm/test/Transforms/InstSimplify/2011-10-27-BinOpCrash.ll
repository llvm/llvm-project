; RUN: opt < %s -instcombine

@_ZN11xercesc_2_5L11gDigitCharsE = external constant [32 x i16], align 2
@_ZN11xercesc_2_5L10gBaseCharsE = external constant [354 x i16], align 2
@_ZN11xercesc_2_5L17gIdeographicCharsE = external constant [7 x i16], align 2
@_ZN11xercesc_2_5L15gCombiningCharsE = external constant [163 x i16], align 2

define i32 @_ZN11xercesc_2_515XMLRangeFactory11buildRangesEv(i32 %x) {
  %a = add i32 %x, add (i32 add (i32 ashr (i32 add (i32 mul (i32 ptrtoint ([32 x i16]* @_ZN11xercesc_2_5L11gDigitCharsE to i32), i32 -1), i32 ptrtoint (i16* getelementptr inbounds ([32 x i16], [32 x i16]* @_ZN11xercesc_2_5L11gDigitCharsE, i32 0, i32 30) to i32)), i32 1), i32 ashr (i32 add (i32 mul (i32 ptrtoint ([7 x i16]* @_ZN11xercesc_2_5L17gIdeographicCharsE to i32), i32 -1), i32 ptrtoint (i16* getelementptr inbounds ([7 x i16], [7 x i16]* @_ZN11xercesc_2_5L17gIdeographicCharsE, i32 0, i32 4) to i32)), i32 1)), i32 8)
  %b = add i32 %a, %x
  ret i32 %b
}
