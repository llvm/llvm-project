# RUN: llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s

# Test basic .inline_asm_mode directive functionality

.text

# Test that the directive is parsed correctly
.inline_asm_mode strict
.inline_asm_mode relaxed

# Test strict mode warnings
.inline_asm_mode strict

# This should produce a warning
# CHECK: warning: non-local label 'unsafe_global' in inline assembly strict mode may be unsafe for external jumps; consider using local labels (.L*) instead
unsafe_global:
    nop

# This should not warn (local label)
.L_safe_local:
    nop

# Test error handling
.inline_asm_mode invalid
# CHECK: error: expected 'strict' or 'relaxed'