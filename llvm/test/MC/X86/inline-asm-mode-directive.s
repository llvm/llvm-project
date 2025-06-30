# RUN: llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s

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


# Test strict mode - should warn about non-numeric labels
.inline_asm_mode strict

# Only numeric labels are safe - should not warn
1:
    nop
42:
    nop
999:
    nop

# All other labels should warn
# CHECK: :[[@LINE+1]]:1: warning: non-numeric label '.L_local1' in inline assembly strict mode may be unsafe for external jumps; consider using numeric labels (1:, 2:, etc.) instead
.L_local1:
    nop

# CHECK: :[[@LINE+1]]:1: warning: non-numeric label '.L_local_with_numbers_123' in inline assembly strict mode may be unsafe for external jumps; consider using numeric labels (1:, 2:, etc.) instead
.L_local_with_numbers_123:
    nop

# CHECK: :[[@LINE+1]]:1: warning: non-numeric label 'unsafe_label' in inline assembly strict mode may be unsafe for external jumps; consider using numeric labels (1:, 2:, etc.) instead
unsafe_label:
    nop

# CHECK: :[[@LINE+1]]:1: warning: non-numeric label 'another_global' in inline assembly strict mode may be unsafe for external jumps; consider using numeric labels (1:, 2:, etc.) instead
another_global:
    nop

# Switch back to relaxed mode
.inline_asm_mode relaxed

# This should not warn again
yet_another_label:
    nop


