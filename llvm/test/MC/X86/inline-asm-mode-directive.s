# RUN: llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=RELAX
# RUN: llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=STRICT

# Test the .inline_asm_mode directive for safer inline assembly label handling

.text

# Test relaxed mode (default) - no warnings
.inline_asm_mode relaxed

# These labels should not produce warnings in relaxed mode
my_label:
    nop
global_symbol:
    nop
.L_local_label:
    nop

# RELAX-NOT: warning

# Test strict mode - should warn about non-local labels
.inline_asm_mode strict

# Local labels - should not warn
.L_local1:
    nop
.L_local_with_numbers_123:
    nop
1:
    nop
42:
    nop

# Non-local labels - should warn
# STRICT: :[[@LINE+1]]:1: warning: non-local label 'unsafe_label' in inline assembly strict mode may be unsafe for external jumps; consider using local labels (.L*) instead
unsafe_label:
    nop

# STRICT: :[[@LINE+1]]:1: warning: non-local label 'another_global' in inline assembly strict mode may be unsafe for external jumps; consider using local labels (.L*) instead
another_global:
    nop

# Switch back to relaxed mode
.inline_asm_mode relaxed

# This should not warn again
yet_another_label:
    nop

# RELAX-NOT: warning

# Test error cases
.inline_asm_mode invalid_mode
# CHECK: :[[@LINE-1]]:18: error: expected 'strict' or 'relaxed'

.inline_asm_mode
# CHECK: :[[@LINE-1]]:17: error: expected 'strict' or 'relaxed' after '.inline_asm_mode'