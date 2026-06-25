; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: declare rangeset(i32 (0, 2), (5, 8)) i32 @ret_rangeset()
declare rangeset(i32 (0, 2), (5, 8)) i32 @ret_rangeset()

; CHECK: declare void @param_rangeset(i32 rangeset(i32 (0, 2), (5, 8)))
declare void @param_rangeset(i32 rangeset(i32 (0, 2), (5, 8)))

; CHECK: declare rangeset(i32 (0, 2), (5, 8)) <4 x i32> @ret_rangeset_vec()
declare rangeset(i32 (0, 2), (5, 8)) <4 x i32> @ret_rangeset_vec()

; CHECK: declare void @param_rangeset_vec(<4 x i32> rangeset(i32 (0, 2), (5, 8)))
declare void @param_rangeset_vec(<4 x i32> rangeset(i32 (0, 2), (5, 8)))

; CHECK: declare rangeset(i32 (1, 2147483647)) i32 @ret_rangeset_terminal()
declare rangeset(i32 (1, 2147483647)) i32 @ret_rangeset_terminal()

; CHECK: declare rangeset(i32 (0, 4), (6, 8)) i32 @ret_rangeset_coalesced()
declare rangeset(i32 (0, 2), (3, 4), (6, 8)) i32 @ret_rangeset_coalesced()
