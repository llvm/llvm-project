##===- examples/OCaml-Kaleidoscope/Chapter6/Makefile -------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
# 
# This is the makefile for the Objective Caml kaleidoscope tutorial, chapter 6.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../../..
TOOLNAME := OCaml-Kaleidoscope-Ch6
EXAMPLE_TOOL := 1
UsedComponents := core
UsedOcamLibs := llvm llvm_analysis llvm_executionengine llvm_target \
	llvm_scalar_opts

OCAMLCFLAGS += -pp camlp4of

OcamlSources1 = \
	$(PROJ_SRC_DIR)/ast.ml \
	$(PROJ_SRC_DIR)/parser.ml \
	$(PROJ_SRC_DIR)/codegen.ml \
	$(PROJ_SRC_DIR)/lexer.ml \
	$(PROJ_SRC_DIR)/token.ml \
	$(PROJ_SRC_DIR)/toplevel.ml \
	$(PROJ_SRC_DIR)/toy.ml

ExcludeSources = $(PROJ_SRC_DIR)/myocamlbuild.ml

include $(LEVEL)/bindings/ocaml/Makefile.ocaml
