..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx12_attr:

attr
====

Interpolation attribute and channel:

    ============== ===================================
    Syntax         Description
    ============== ===================================
    attr{0..32}.x  Attribute 0..32 with *x* channel.
    attr{0..32}.y  Attribute 0..32 with *y* channel.
    attr{0..32}.z  Attribute 0..32 with *z* channel.
    attr{0..32}.w  Attribute 0..32 with *w* channel.
    ============== ===================================

Examples:

.. parsed-literal::

    ds_param_load v1, attr0.x
