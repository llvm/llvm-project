set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  )

add_llvm_unittest(BitcodeTests
  BitReaderTest.cpp
  )
