set(LLVM_TARGET_DEFINITIONS Attributes.td)
tablegen(LLVM Attributes_renamed.inc -gen-attrs)

set(LLVM_TARGET_DEFINITIONS Intrinsics.td)
tablegen(LLVM Intrinsics.gen -gen-intrinsic)
add_public_tablegen_target(intrinsics_gen)
