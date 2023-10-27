
    README for the OpenMP Tooling Interface Testing Library (libomptest)
    ====================================================================

Introduction
============
TBD

Testing macros
==============

## OMPT_ASSERT_SEQUENCE
This macro will check for the occurrence of the provided event, which also
entails the exact sequence of events. When only using this assertion macro one
has to provide every single event in the exact order of occurrence.

## OMPT_ASSERT_SEQUENCE_ONLY
This macro will act like `OMPT_ASSERT_SEQUENCE`, while being preceded
-AND- succeeded by commands to suspend sequenced assertion until the next match.
As a result, one may omit all other "unneccessary" events from the sequence.

## OMPT_ASSERT_NAMED_SEQUENCE
This macro will act like `OMPT_ASSERT_SEQUENCE` but internally the provided name
will be stored within the created event and may be printed in messages.

## OMPT_ASSERT_GROUPED_SEQUENCE
This macro will act like `OMPT_ASSERT_SEQUENCE` with the addition of grouping.

## OMPT_ASSERT_GROUPED_SEQUENCE_ONLY
This macro will act like `OMPT_ASSERT_SEQUENCE_ONLY`, plus grouping.

### Grouping Asserts

This allows to generate and verify data during runtime of a test.
Currently, we only use target region information which manifests into groups.
This allows to correlate multiple events to a certain target region without
manual interaction just by specifying a groupname for these events.

When a target region is encountered and we are about to enter it, we will gather
the `target_id` (non-EMI) -OR- `target_data->value` (EMI). This value is stored
along the groupname for future reference. Upon target region end, we will erase
the corresponding group. (Note: This will make the groupname available again.)

Other asserted callbacks which may occur within target regions, will query their
groupname: retrieving and verifying the value of the group vs. the observed
event's own value.

### Suspending Sequenced Asserts

When a sequence of events is not of interest while testing, these additional
events may be ignored by suspending the assertion until the next match. This
can be done by using `OMPT_ASSERT_SEQUENCE_SUSPEND` manually or the `_ONLY`
macro variants, like `OMPT_ASSERT_GROUPED_SEQUENCE_ONLY`.

The former will add a special event to the queue of expected events and signal
that any non-matching event should be ignored rather than failing the test.
`_ONLY` macros will embed their corresponding macro between two calls to
`OMPT_ASSERT_SEQUENCE_SUSPEND`. As a consequence, we will enter passive
assertion until a match occurs, then enter passive assertion again. This enables
us to "only" assert a ceratin, single event in arbitrary circumstances.

## Shorthands
To allow for easier writing of tests and enhanced readability, the following set
of aliases is introduced. The left hand side represents the original value,
while the right hand side depicts the shorthand version.

  |=================================================================|
  | ompt_scope_endpoint_t                                           |
  |-----------------------------------------------------------------|
  | ompt_scope_begin                            | BEGIN             |
  | ompt_scope_end                              | END               |
  | ompt_scope_beginend                         | BEGINEND          |
  |=================================================================|
  | ompt_target_t                                                   |
  |-----------------------------------------------------------------|
  | ompt_target                                 | TARGET            |
  | ompt_target_enter_data                      | ENTER_DATA        |
  | ompt_target_exit_data                       | EXIT_DATA         |
  | ompt_target_update                          | UPDATE            |
  | ompt_target_nowait                          | TARGET_NOWAIT     |
  | ompt_target_enter_data_nowait               | ENTER_DATA_NOWAIT |
  | ompt_target_exit_data_nowait                | EXIT_DATA_NOWAIT  |
  | ompt_target_update_nowait                   | UPDATE_NOWAIT     |
  |=================================================================|
  | ompt_target_data_op_t                                           |
  |-----------------------------------------------------------------|
  | ompt_target_data_alloc                      | ALLOC             |
  | ompt_target_data_transfer_to_device         | H2D               |
  | ompt_target_data_transfer_from_device       | D2H               |
  | ompt_target_data_delete                     | DELETE            |
  | ompt_target_data_associate                  | ASSOCIATE         |
  | ompt_target_data_disassociate               | DISASSOCIATE      |
  | ompt_target_data_alloc_async                | ALLOC_ASYNC       |
  | ompt_target_data_transfer_to_device_async   | H2D_ASYNC         |
  | ompt_target_data_transfer_from_device_async | D2H_ASYNC         |
  | ompt_target_data_delete_async               | DELETE_ASYNC      |
  |=================================================================|

Limitations
===========
Currently, there are some peculiarities which have to be kept in mind when using
this library:

## Callbacks
  * It is not possible to e.g. test non-EMI -AND- EMI callbacks within the same
    test file. Reason: all testsuites will share the initialization and
    therefore the registered callbacks.
  * It is not possible to check for device initialization and/or load callbacks
    more than once per test file. The first testcase being run, triggering these
    will be the only testcase that is able to check for these callbacks. This is
    because after that, the device remains initialized.
  * It is not possible to check for device finalization callbacks, as libomptest
    is un-loaded before this callback occurs. Same holds true for the final
    ThreadEnd event(s).

Miscellaneous
=============

## Default values

To allow for easier writing of tests, many OMPT events may be created using less
parameters than actually requested by the spec -- by using default values. These
defaults are currently set to the corresponding data type's minimum as follows,
for example integers will use: `std::numeric_limits<int>::min()`.

When an expected / user-specified event has certain values set to the
corresponding default, these values will be ignored. That is, when compared to
an observed event, this property will be considered as 'equal' regardless of
their actual equality relation.

