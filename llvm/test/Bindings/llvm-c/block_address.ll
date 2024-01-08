; RUN: llvm-as < %s | llvm-c-test --module-list-global-block-address-values | FileCheck %s


define void @test_block_address_01() {
entry:
  br label %block_0
block_0:
  ret void
}

define void @test_block_address_02() {
entry:
  br label %block_0
block_0:
  ret void
}

define void @test_block_address_03() {
entry:
  br label %block_0
block_0:
  br label %block_1
block_1:
  ret void
}


@g_block_address_01 = global ptr blockaddress(@test_block_address_01, %block_0)
;CHECK: BlockAddress 'g_block_address_01' Func 'test_block_address_01' Basic Block 'block_0'

@g_block_address_02 = global ptr blockaddress(@test_block_address_02, %block_0)
;CHECK: BlockAddress 'g_block_address_02' Func 'test_block_address_02' Basic Block 'block_0'

@g_block_address_03 = global ptr blockaddress(@test_block_address_03, %block_0)
;CHECK: BlockAddress 'g_block_address_03' Func 'test_block_address_03' Basic Block 'block_0'

@g_block_address_04 = global ptr blockaddress(@test_block_address_03, %block_1)
;CHECK: BlockAddress 'g_block_address_04' Func 'test_block_address_03' Basic Block 'block_1'

