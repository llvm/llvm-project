..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx1030_vaddr_49d53a:

vaddr
=====

Image address which includes from one to four dimensional coordinates and other data used to locate a position in the image.

This operand may be specified using either :ref:`standard VGPR syntax<amdgpu_synid_v>` or special :ref:`NSA VGPR syntax<amdgpu_synid_nsa>`.

*Size:* 8-12 dwords. Actual size depends on :ref:`a16<amdgpu_synid_a16>`.

* If specified using :ref:`NSA VGPR syntax<amdgpu_synid_nsa>`, the size is 8-12 dwords.
* If specified using :ref:`standard VGPR syntax<amdgpu_synid_v>`, the size is 8 dwords. Opcodes which require more than 8 dwords for address size must specify 16 dwords due to a limited range of supported register sequences.

  Examples:

  .. parsed-literal::

    image_bvh_intersect_ray v[4:7], v[9:24], s[4:7]
    image_bvh_intersect_ray v[39:42], [v5, v4, v2, v1, v7, v3, v0, v6], s[12:15] a16

*Operands:* :ref:`v<amdgpu_synid_v>`
