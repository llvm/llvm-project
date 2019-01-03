..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid8_msg:

msg
===========================

A 16-bit message code. The bits of this operand have the following meaning:

    ============ ======================================================
    Bits         Description
    ============ ======================================================
    3:0          Message *type*.
    6:4          Optional *operation*.
    9:7          Optional *parameters*.
    15:10        Unused.
    ============ ======================================================

This operand may be specified as a positive 16-bit :ref:`integer_number<amdgpu_synid_integer_number>` or using the syntax described below:

    ======================================== ========================================================================
    Syntax                                   Description
    ======================================== ========================================================================
    sendmsg(<*type*>)                        A message identified by its *type*.
    sendmsg(<*type*>, <*op*>)                A message identified by its *type* and *operation*.
    sendmsg(<*type*>, <*op*>, <*stream*>)    A message identified by its *type* and *operation* with a stream *id*.
    ======================================== ========================================================================

*Type* may be specified using message *name* or message *id*.

*Op* may be specified using operation *name* or operation *id*.

Stream *id* is an integer in the range 0..3.

Message *id*, operation *id* and stream *id* must be specified as positive :ref:`integer numbers<amdgpu_synid_integer_number>`.

Each message type supports specific operations:

    ================= ========== ============================== ============ ==========
    Message name      Message Id Supported Operations           Operation Id Stream Id
    ================= ========== ============================== ============ ==========
    MSG_INTERRUPT     1          \-                             \-           \-
    MSG_GS            2          GS_OP_CUT                      1            Optional
    \                            GS_OP_EMIT                     2            Optional
    \                            GS_OP_EMIT_CUT                 3            Optional
    MSG_GS_DONE       3          GS_OP_NOP                      0            \-
    \                            GS_OP_CUT                      1            Optional
    \                            GS_OP_EMIT                     2            Optional
    \                            GS_OP_EMIT_CUT                 3            Optional
    MSG_SYSMSG        15         SYSMSG_OP_ECC_ERR_INTERRUPT    1            \-
    \                            SYSMSG_OP_REG_RD               2            \-
    \                            SYSMSG_OP_HOST_TRAP_ACK        3            \-
    \                            SYSMSG_OP_TTRACE_PC            4            \-
    ================= ========== ============================== ============ ==========

Examples:

.. parsed-literal::

    s_sendmsg 0x12
    s_sendmsg sendmsg(MSG_INTERRUPT)
    s_sendmsg sendmsg(2, GS_OP_CUT)
    s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT)
    s_sendmsg sendmsg(MSG_GS, 2)
    s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_EMIT_CUT, 1)
    s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_TTRACE_PC)

