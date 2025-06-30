# RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s

# Test error handling for .inline_asm_mode directive

.text

# Test invalid mode
.inline_asm_mode invalid
# CHECK: error: expected 'strict' or 'relaxed'

# Test missing mode
.inline_asm_mode
# CHECK: error: expected 'strict' or 'relaxed' after '.inline_asm_mode'