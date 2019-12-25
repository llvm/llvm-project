..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

====================================================================================
Syntax of gfx900, gfx902 and gfx909 Instructions
====================================================================================

.. contents::
  :local:

Introduction
============

This document describes the syntax of *instructions specific to gfx900, gfx902 and gfx909*.

For a description of other gfx900, gfx902 and gfx909 instructions see :doc:`Syntax of Core GFX9 Instructions<AMDGPUAsmGFX9>`.

Notation
========

Notation used in this document is explained :ref:`here<amdgpu_syn_instruction_notation>`.

Overvew
=======

An overview of generic syntax and other features of AMDGPU instructions may be found :ref:`in this document<amdgpu_syn_instructions>`.

Instructions
============


VOP3P
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**       **SRC1**       **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_mad_mix_f32                  :ref:`vdst<amdgpu_synid900_vdst32_0>`,     :ref:`src0<amdgpu_synid900_src32_0>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`, :ref:`src1<amdgpu_synid900_src32_1>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`, :ref:`src2<amdgpu_synid900_src32_1>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`      :ref:`m_op_sel<amdgpu_synid_mad_mix_op_sel>` :ref:`m_op_sel_hi<amdgpu_synid_mad_mix_op_sel_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_mad_mixhi_f16                :ref:`vdst<amdgpu_synid900_vdst32_0>`,     :ref:`src0<amdgpu_synid900_src32_0>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`, :ref:`src1<amdgpu_synid900_src32_1>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`, :ref:`src2<amdgpu_synid900_src32_1>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`      :ref:`m_op_sel<amdgpu_synid_mad_mix_op_sel>` :ref:`m_op_sel_hi<amdgpu_synid_mad_mix_op_sel_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_mad_mixlo_f16                :ref:`vdst<amdgpu_synid900_vdst32_0>`,     :ref:`src0<amdgpu_synid900_src32_0>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`, :ref:`src1<amdgpu_synid900_src32_1>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`, :ref:`src2<amdgpu_synid900_src32_1>`::ref:`m<amdgpu_synid900_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid900_mad_type_dev>`      :ref:`m_op_sel<amdgpu_synid_mad_mix_op_sel>` :ref:`m_op_sel_hi<amdgpu_synid_mad_mix_op_sel_hi>` :ref:`clamp<amdgpu_synid_clamp>`

.. |---| unicode:: U+02014 .. em dash


.. toctree::
    :hidden:

    AMDGPUAsmGFX9
    gfx900_src32_0
    gfx900_src32_1
    gfx900_vdst32_0
    gfx900_mad_type_dev
    gfx900_mod_vop3_abs_neg
