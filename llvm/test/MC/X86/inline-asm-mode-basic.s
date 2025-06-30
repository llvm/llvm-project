# RUN: llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s

# Test basic .inline_asm_mode directive functionality

.text

# Test that the directive is parsed correctly
.inline_asm_mode strict
.inline_asm_mode relaxed

# Test strict mode warnings
.inline_asm_mode strict

# This should produce a warning (non-numeric label)
# CHECK: warning: non-numeric label 'unsafe_global' in inline assembly strict mode may be unsafe for external jumps; consider using numeric labels (1:, 2:, etc.) instead
unsafe_global:
    nop

# This should also warn
# CHECK: warning: non-numeric label '.L_unsafe_local' in inline assembly strict mode may be unsafe for external jumps; consider using numeric labels (1:, 2:, etc.) instead
.L_unsafe_local:
    nop

# No warning
1:
    nop

