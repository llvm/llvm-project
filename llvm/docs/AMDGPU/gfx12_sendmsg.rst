.. _amdgpu_synid_sendmsg:

sendmsg
=======

An 8-bit value in simm16[7:0] encodes the message type.

This operand may be specified as one of the following:

* An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..0xFFFF.
* A *sendmsg* value described below.


    ==================================== ====================================================
    Sendmsg Value Syntax                 Description
    ==================================== ====================================================
    sendmsg(<*type*>)                    A message identified by its *type*.
    ==================================== ====================================================

*Type* may be specified using message *name* or message *id*.

Numeric values may be specified as positive :ref:`integer numbers<amdgpu_synid_integer_number>`
or :ref:`absolute expressions<amdgpu_synid_absolute_expression>`.


Only the following message types are valid.

    ====================== ===========
    Message type           simm16[7:0]
    ====================== ===========
    Reserved               0 
    MSG_INTERRUPT          1
    MSG_HS_TESSFACTOR      2
    MSG_DEALLOC_VGPRS      3
    MSG_GS_ALLOC_REQ       9
    ====================== ===========

Examples:

.. parsed-literal::

    // numeric message code
    msg = 0x1
    s_sendmsg 0x3
    s_sendmsg msg + 2

    // sendmsg with strict arguments validation
    s_sendmsg sendmsg(MSG_INTERRUPT)
