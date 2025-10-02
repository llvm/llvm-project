.. _amdgpu_synid_sendmsg_rtn:

sendmsg_rtn
===========

An 8-bit value in the instruction to encode the message type.

This operand may be specified as one of the following:

    * An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..0xFFFF.
    * A *sendmsg* value described below.

    ==================================== ====================================================
    Sendmsg Value Syntax                 Description
    ==================================== ====================================================
    sendmsg(MSG_RTN_GET_DOORBELL)        Get doorbell ID.
    sendmsg(MSG_RTN_GET_DDID)            Get Draw/Dispatch ID.
    sendmsg(MSG_RTN_GET_TMA)             Get TMA value.
    sendmsg(MSG_RTN_GET_TBA)             Get TBA value.
    sendmsg(MSG_RTN_GET_REALTIME)        Get REALTIME value.
    sendmsg(MSG_RTN_SAVE_WAVE)           Report that this wave is ready to be context-saved.
    ==================================== ====================================================

Examples:

.. parsed-literal::

    s_sendmsg_rtn_b32 s0, 132
    s_sendmsg_rtn_b32 s0, sendmsg(MSG_GET_REALTIME)

