@ RUN: not llvm-mc -triple armv7--none-eabi -filetype asm -o /dev/null 2>&1 %s | FileCheck %s

	.arch_extension os
CHECK: error: unsupported architectural extension: os

	.arch_extension iwmmxt
CHECK: error: unsupported architectural extension: iwmmxt

	.arch_extension iwmmxt2
CHECK: error: unsupported architectural extension: iwmmxt2

	.arch_extension maverick
CHECK: error: unsupported architectural extension: maverick

	.arch_extension xscale
CHECK: error: unsupported architectural extension: xscale

	.arch_extension invalid_extension_name
CHECK: error: unknown architectural extension: invalid_extension_name

	.arch_extension 42
CHECK: error: expected architecture extension name

	.arch_extension
CHECK: error: expected architecture extension name
