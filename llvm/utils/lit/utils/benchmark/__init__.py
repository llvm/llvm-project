# ===------------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#
# WARNING: This is a highly experimental package. Use at your own risk.
"""
Platform assumptions:
  Linux - x86-64 Intel CPU, sudo; use --skip-env-setup if unsupported
  macOS - caffeinate available; no CPU-pinning API
  Windows - x86-64, env setup requires Administrator
"""
