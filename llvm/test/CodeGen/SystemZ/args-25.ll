; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check
; REQUIRES: asserts
;
; Test that the lowering of returns as seen by the instruction selector works
; in accordance to the function header.

define noext i32 @fun_NoExtRet_ccc() { ret i32 0 }
define signext i32 @fun_SignExtRet_ccc() { ret i32 0 }
define zeroext i32 @fun_ZeroExtRet_ccc() { ret i32 0 }

define internal noext i32 @fun_NoExtRet_addrtaken() { ret i32 0 }
define internal signext i32 @fun_SignExtRet_addrtaken() { ret i32 0 }
define internal zeroext i32 @fun_ZeroExtRet_addrtaken() { ret i32 0 }

declare void @ExtFun(ptr %FunPtr)
define void @foo() {
  call void @ExtFun(ptr @fun_NoExtRet_addrtaken)
  call void @ExtFun(ptr @fun_SignExtRet_addrtaken)
  call void @ExtFun(ptr @fun_ZeroExtRet_addrtaken)
  ret void
}

define internal i32 @fun_internal() { ret i32 0 }
define internal noext i32 @fun_NoExtRet_internal() { ret i32 0 }
define internal signext i32 @fun_SignExtRet_internal() { ret i32 0 }
define internal zeroext i32 @fun_ZeroExtRet_internal() { ret i32 0 }

define fastcc i32 @fun_fastcc() { ret i32 0 }
define internal noext i32 @fun_NoExtRet_fastcc() { ret i32 0 }
define internal signext i32 @fun_SignExtRet_fastcc() { ret i32 0 }
define internal zeroext i32 @fun_ZeroExtRet_fastcc() { ret i32 0 }
