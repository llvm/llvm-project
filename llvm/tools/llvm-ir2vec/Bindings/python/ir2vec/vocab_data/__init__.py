# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This directory is intentionally empty in the repository.
#
# The vocabulary JSON file (seedEmbeddingVocab75D.json) lives at:
#   llvm/lib/Analysis/models/seedEmbeddingVocab75D.json
#
# It is injected into this directory at wheel build time by the build script
# (buildscripts/build_wheel_local.sh), so the assembled wheel is self-contained.
# Do not add JSON files here directly — the Analysis models directory is the
# single source of truth.