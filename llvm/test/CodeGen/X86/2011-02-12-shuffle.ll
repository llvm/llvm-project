; RUN: llc < %s
; PR9165

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i686-pc-win32"

define void @m_387() nounwind {
entry:
  br i1 undef, label %if.end, label %UnifiedReturnBlock

if.end:                                           ; preds = %entry
  %tmp1067 = load <16 x i32> addrspace(1)* null, align 64
  %tmp1082 = shufflevector         <16 x i32> <i32 0, i32 0, i32 0, i32 undef, i32 undef, i32 0, i32 0, i32 undef, i32 0, i32 0, i32 undef, i32 undef, i32 0, i32 undef, i32 undef, i32 undef>, 
                                                                                                                <16 x i32> %tmp1067, 
                                                                                                                <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 26, i32 5, i32 6, i32 undef, i32 8, i32 9, i32 31, i32 30, i32 12, i32 undef, i32 undef, i32 undef>
  
  %tmp1100 = shufflevector         <16 x i32> %tmp1082, 
                                                                                                                <16 x i32> %tmp1067, 
                                                                                                                <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 4, i32 5, i32 6, i32 18, i32 8, i32 9, i32 10, i32 11, i32 12, i32 25, i32 undef, i32 17>
  
  %tmp1112 = shufflevector         <16 x i32> %tmp1100, 
                                                                                                                <16 x i32> %tmp1067, 
                                                                                                                <16 x i32> <i32 0, i32 1, i32 2, i32 24, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 18, i32 15>
  
  store <16 x i32> %tmp1112, <16 x i32> addrspace(1)* undef, align 64
  
  ret void

UnifiedReturnBlock:                               ; preds = %entry
  ret void
}

