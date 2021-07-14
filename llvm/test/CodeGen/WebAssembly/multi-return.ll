; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Return multiple values, some of which will be legalized into multiple values.
declare { i64, i128, i192, i128, i64 } @return_multi_multi()

; Test returning a single value from @return_multi_multi.

define i64 @test0() {
; CHECK-LABEL: test0
; CHECK: call    	return_multi_multi
; CHECK: i64.load	$[[RV:[0-9]+]]=, 8(${{[0-9]+}})
; CHECK: local.copy	$push8=, $[[RV]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %t1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 0
  ret i64 %t1
}

define i128 @test1() {
; CHECK-LABEL: test1
; CHECK: call    	return_multi_multi
; CHECK: i64.load	$[[RV:[0-9]+]]=, 16($[[SP:[0-9]+]])
; CHECK: i32.const	$push0=, 24
; CHECK: i32.add 	$push1=, $[[SP]], $pop0
; CHECK: i64.load	$push2=, 0($pop1)
; CHECK: i64.store	8($0), $pop2
; CHECK: i64.store	0($0), $[[RV]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %t1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 1
  ret i128 %t1
}

define i192 @test2() {
; CHECK-LABEL: test2
; CHECK: call    	return_multi_multi
; CHECK: i32.const	$push0=, 40
; CHECK: i32.add 	$push1=, $[[SP:[0-9]+]], $pop0
; CHECK: i64.load	$[[L1:[0-9]+]]=, 0($pop1)
; CHECK: i64.load	$[[L2:[0-9]+]]=, 32($[[SP]])
; CHECK: i32.const	$push2=, 48
; CHECK: i32.add 	$push3=, $[[SP]], $pop2
; CHECK: i64.load	$push4=, 0($pop3)
; CHECK: i64.store	16($0), $pop4
; CHECK: i64.store	0($0), $[[L2]]
; CHECK: i64.store	8($0), $[[L1]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %t1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 2
  ret i192 %t1
}

define i128 @test3() {
; CHECK-LABEL: test3
; CHECK: call    	return_multi_multi
; CHECK: i64.load	$[[L1:[0-9]+]]=, 56($[[SP:[0-9]+]])
; CHECK: i32.const	$push0=, 64
; CHECK: i32.add 	$push1=, $[[SP]], $pop0
; CHECK: i64.load	$push2=, 0($pop1)
; CHECK: i64.store	8($0), $pop2
; CHECK: i64.store	0($0), $[[L1]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %t1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 3
  ret i128 %t1
}

define i64 @test4() {
; CHECK-LABEL: test4
; CHECK: call    	return_multi_multi
; CHECK: i64.load	$[[L1:[0-9]+]]=, 72($[[SP:[0-9]+]])
; CHECK: local.copy	$push8=, $[[L1]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %t1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 4
  ret i64 %t1
}

; Test returning multiple values from @return_multi_multi.

define { i64, i128 } @test5() {
; CHECK-LABEL: test5
; CHECK: call    	return_multi_multi
; CHECK: i32.const	$push10=, 8
; CHECK: i32.add 	$push11=, $[[SP:[0-9]+]], $pop10
; CHECK: i32.const	$push0=, 16
; CHECK: i32.add 	$push1=, $pop11, $pop0
; CHECK: i64.load	$[[L1:[0-9]+]]=, 0($pop1)
; CHECK: i64.load	$[[L2:[0-9]+]]=, 8($[[SP]])
; CHECK: i64.load	$push2=, 16($[[SP]])
; CHECK: i64.store	8($0), $pop2
; CHECK: i32.const	$push12=, 16
; CHECK: i32.add 	$push3=, $0, $pop12
; CHECK: i64.store	0($pop3), $[[L1]]
; CHECK: i64.store	0($0), $[[L2]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %r0 = extractvalue { i64, i128, i192, i128, i64 } %t0, 0
  %r1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 1
  %s0 = insertvalue { i64, i128 } undef, i64 %r0, 0
  %s1 = insertvalue { i64, i128 } %s0, i128 %r1, 1
  ret { i64, i128 } %s1
}

define { i128, i128 } @test6() {
; CHECK-LABEL: test6
; CHECK: call    	return_multi_multi
; CHECK: i32.const	$push0=, 24
; CHECK: i32.add 	$push1=, $[[SP:[0-9]+]], $pop0
; CHECK: i64.load	$[[L1:[0-9]+]]=, 0($pop1)
; CHECK: i32.const	$push2=, 64
; CHECK: i32.add 	$push3=, $[[SP]], $pop2
; CHECK: i64.load	$[[L2:[0-9]+]]=, 0($pop3)
; CHECK: i64.load	$[[L3:[0-9]+]]=, 16($[[SP]])
; CHECK: i64.load	$push4=, 56($[[SP]])
; CHECK: i64.store	16($0), $pop4
; CHECK: i32.const	$push5=, 24
; CHECK: i32.add 	$push6=, $0, $pop5
; CHECK: i64.store	0($pop6), $[[L2]]
; CHECK: i64.store	0($0), $[[L3]]
; CHECK: i64.store	8($0), $[[L1]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %r1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 1
  %r3 = extractvalue { i64, i128, i192, i128, i64 } %t0, 3
  %s0 = insertvalue { i128, i128 } undef, i128 %r1, 0
  %s1 = insertvalue { i128, i128 } %s0, i128 %r3, 1
  ret { i128, i128 } %s1
}

define { i64, i192 } @test7() {
; CHECK-LABEL: test7
; CHECK: call    	return_multi_multi
; CHECK: i32.const	$push2=, 40
; CHECK: i32.add 	$push3=, $[[SP:[0-9]+]], $pop2
; CHECK: i64.load	$[[L1:[0-9]+]]=, 0($pop3)
; CHECK: i64.load	$[[L2:[0-9]+]]=, 8($[[SP]])
; CHECK: i64.load	$[[L3:[0-9]+]]=, 32($[[SP]])
; CHECK: i32.const	$push0=, 24
; CHECK: i32.add 	$push1=, $0, $pop0
; CHECK: i32.const	$push4=, 48
; CHECK: i32.add 	$push5=, $[[SP]], $pop4
; CHECK: i64.load	$push6=, 0($pop5)
; CHECK: i64.store	0($pop1), $pop6
; CHECK: i64.store	8($0), $[[L3]]
; CHECK: i32.const	$push7=, 16
; CHECK: i32.add 	$push8=, $0, $pop7
; CHECK: i64.store	0($pop8), $[[L1]]
; CHECK: i64.store	0($0), $[[L2]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %r0 = extractvalue { i64, i128, i192, i128, i64 } %t0, 0
  %r2 = extractvalue { i64, i128, i192, i128, i64 } %t0, 2
  %s0 = insertvalue { i64, i192 } undef, i64 %r0, 0
  %s1 = insertvalue { i64, i192 } %s0, i192 %r2, 1
  ret { i64, i192 } %s1
}

define { i128, i192, i128, i64 } @test8() {
; CHECK-LABEL: test8
; CHECK: call    	return_multi_multi
; CHECK: i32.const	$push0=, 64
; CHECK: i32.add 	$push1=, $[[SP:[0-9]+]], $pop0
; CHECK: i64.load	$[[L1:[0-9]+]]=, 0($pop1)
; CHECK: i32.const	$push20=, 8
; CHECK: i32.add 	$push21=, $[[SP]], $pop20
; CHECK: i32.const	$push2=, 32
; CHECK: i32.add 	$push3=, $pop21, $pop2
; CHECK: i64.load	$[[L2:[0-9]+]]=, 0($pop3)
; CHECK: i32.const	$push4=, 48
; CHECK: i32.add 	$push5=, $[[SP]], $pop4
; CHECK: i64.load	$[[L3:[0-9]+]]=, 0($pop5)
; CHECK: i32.const	$push6=, 24
; CHECK: i32.add 	$push7=, $[[SP]], $pop6
; CHECK: i64.load	$[[L4:[0-9]+]]=, 0($pop7)
; CHECK: i64.load	$[[L5:[0-9]+]]=, 8($[[SP]])
; CHECK: i64.load	$[[L6:[0-9]+]]=, 56($[[SP]])
; CHECK: i64.load	$[[L7:[0-9]+]]=, 32($[[SP]])
; CHECK: i64.load	$push8=, 16($[[SP]])
; CHECK: i64.store	40($0), $pop8
; CHECK: i32.const	$push9=, 48
; CHECK: i32.add 	$push10=, $0, $pop9
; CHECK: i64.store	0($pop10), $[[L4]]
; CHECK: i32.const	$push22=, 32
; CHECK: i32.add 	$push11=, $0, $pop22
; CHECK: i64.store	0($pop11), $[[L3]]
; CHECK: i64.store	16($0), $[[L7]]
; CHECK: i32.const	$push12=, 24
; CHECK: i32.add 	$push13=, $0, $pop12
; CHECK: i64.store	0($pop13), $[[L2]]
; CHECK: i64.store	0($0), $[[L6]]
; CHECK: i64.store	8($0), $[[L1]]
; CHECK: i64.store	56($0), $[[L5]]
  %t0 = call { i64, i128, i192, i128, i64 } @return_multi_multi()
  %r0 = extractvalue { i64, i128, i192, i128, i64 } %t0, 0
  %r1 = extractvalue { i64, i128, i192, i128, i64 } %t0, 1
  %r2 = extractvalue { i64, i128, i192, i128, i64 } %t0, 2
  %r3 = extractvalue { i64, i128, i192, i128, i64 } %t0, 3
  %s0 = insertvalue { i128, i192, i128, i64 } undef, i128 %r3, 0
  %s1 = insertvalue { i128, i192, i128, i64 } %s0, i192 %r2, 1
  %s2 = insertvalue { i128, i192, i128, i64 } %s1, i128 %r1, 2
  %s3 = insertvalue { i128, i192, i128, i64 } %s2, i64 %r0, 3
  ret { i128, i192, i128, i64 } %s3
}
