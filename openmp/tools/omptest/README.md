
README for the OpenMP Tooling Interface Testing Library (ompTest)
=================================================================

# Introduction
OpenMP Tooling Interface Testing Library (ompTest)
ompTest is a unit testing framework for testing OpenMP implementations.
It offers a simple-to-use framework that allows a tester to check for OMPT
events in addition to regular unit testing code, supported by linking against
GoogleTest by default. It also facilitates writing concise tests while bridging
the semantic gap between the unit under test and the OMPT-event testing.

# Testing macros

Corresponding macro definitions are located in:  `./include/AssertMacros.h`

## OMPT_GENERATE_EVENTS(NumberOfCopies, EventMacro)
`TODO`

## OMPT_ASSERT_SET_EVENT(Name, Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SET(EventTy, ...)
`TODO`

## OMPT_ASSERT_SET_GROUPED(Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SET_NAMED(Name, EventTy, ...)
`TODO`

## OMPT_ASSERT_SET_EVENT_NOT(Name, Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SET_NOT(EventTy, ...)
`TODO`

## OMPT_ASSERT_SET_GROUPED_NOT(Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SET_NAMED_NOT(Name, EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE_EVENT(Name, Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE(EventTy, ...)
This macro checks for the occurrence of the provided event, which also
entails the exact sequence of events. When only using this assertion macro one
has to provide every single event in the exact order of occurrence.

## OMPT_ASSERT_SEQUENCE_GROUPED(Group, EventTy, ...)
This macro acts like `OMPT_ASSERT_SEQUENCE` with the addition of grouping.

## OMPT_ASSERT_SEQUENCE_NAMED(Name, EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE_EVENT_NOT(Name, Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE_NOT(EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE_GROUPED_NOT(Group, EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE_NAMED_NOT(Name, EventTy, ...)
`TODO`

## OMPT_ASSERT_SEQUENCE_SUSPEND()
`TODO`

## OMPT_ASSERT_SEQUENCE_ONLY(EventTy, ...)
This macro acts like `OMPT_ASSERT_SEQUENCE`, while actually being preceded
-AND- succeeded by commands to suspend sequenced assertion until the next match.
As a result, one may omit all other "unnecessary" events from the sequence.

## OMPT_ASSERT_SEQUENCE_GROUPED_ONLY(Group, EventTy, ...)
This macro acts like `OMPT_ASSERT_SEQUENCE_ONLY`, plus grouping.

## OMPT_ASSERT_SEQUENCE_NAMED_ONLY(Name, EventTy, ...)
`TODO`

## OMPT_ASSERTER_MODE_STRICT(Asserter)
`TODO`

## OMPT_ASSERTER_MODE_RELAXED(Asserter)
`TODO`

## OMPT_ASSERT_SEQUENCE_MODE_STRICT()
`TODO`

## OMPT_ASSERT_SEQUENCE_MODE_RELAXED()
`TODO`

## OMPT_ASSERT_SET_MODE_STRICT()
`TODO`

## OMPT_ASSERT_SET_MODE_RELAXED()
`TODO`

## OMPT_ASSERTER_DISABLE(Asserter)
`TODO`

## OMPT_ASSERTER_ENABLE(Asserter)
`TODO`

## OMPT_ASSERT_SET_DISABLE()
`TODO`

## OMPT_ASSERT_SET_ENABLE()
`TODO`

## OMPT_ASSERT_SEQUENCE_DISABLE()
`TODO`

## OMPT_ASSERT_SEQUENCE_ENABLE()
`TODO`

## OMPT_REPORT_EVENT_DISABLE()
`TODO`

## OMPT_REPORT_EVENT_ENABLE()
`TODO`

## OMPT_ASSERTER_PERMIT_EVENT(Asserter, EventTy)
`TODO`

## OMPT_ASSERTER_SUPPRESS_EVENT(Asserter, EventTy)
`TODO`

## OMPT_PERMIT_EVENT(EventTy)
`TODO`

## OMPT_SUPPRESS_EVENT(EventTy)
`TODO`

## OMPT_ASSERTER_LOG_LEVEL(Asserter, LogLevel)
`TODO`

## OMPT_ASSERTER_LOG_FORMATTED(Asserter, FormatLog)
`TODO`

## OMPT_ASSERT_SYNC_POINT(SyncPointName)
`TODO`

### Grouping Asserts

This allows to generate and verify data during runtime of a test.
Currently, we only use target region information which manifests into groups.
This allows to correlate multiple events to a certain target region without
manual interaction just by specifying a groupname for these events.

When a target region is encountered and we are about to enter it, we gather the
`target_id` (non-EMI) -OR- `target_data->value` (EMI). This value is stored
along the groupname for future reference. Upon target region end, the
corresponding group is erased. (Note: The groupname is available again.)

Other asserted callbacks which may occur within target regions query their
groupname: retrieving and comparing the value of the group against the observed
event's value.

### Suspending Sequenced Asserts

When a sequence of events is not of interest while testing, these additional
events may be ignored by suspending the assertion until the next match. This
can be done by using `OMPT_ASSERT_SEQUENCE_SUSPEND` manually or the `_ONLY`
macro variants, like `OMPT_ASSERT_GROUPED_SEQUENCE_ONLY`.

The former adds a special event to the queue of expected events and signal
that any non-matching event should be ignored rather than failing the test.
`_ONLY` macros embed their corresponding macro between two calls to
`OMPT_ASSERT_SEQUENCE_SUSPEND`. As a consequence, we enter passive assertion
until a match occurs, then enter passive assertion again. This enables us to
"only" assert a certain, single event in arbitrary circumstances.

### Asserter Modes
`TODO`

## Aliases (shorthands)
To allow for easier writing of tests and enhanced readability, the following set
of aliases is introduced. The left hand side represents the original value,
while the right hand side depicts the shorthand version.

| Type                      | Enum Value                                  | Shorthand                 |
|---------------------------|---------------------------------------------|---------------------------|
| **ompt_scope_endpoint_t** |                                             |                           |
|                           | ompt_scope_begin                            | BEGIN                     |
|                           | ompt_scope_end                              | END                       |
|                           | ompt_scope_beginend                         | BEGINEND                  |
| **ompt_target_t**         |                                             |                           |
|                           | ompt_target                                 | TARGET                    |
|                           | ompt_target_enter_data                      | ENTER_DATA                |
|                           | ompt_target_exit_data                       | EXIT_DATA                 |
|                           | ompt_target_update                          | UPDATE                    |
|                           | ompt_target_nowait                          | TARGET_NOWAIT             |
|                           | ompt_target_enter_data_nowait               | ENTER_DATA_NOWAIT         |
|                           | ompt_target_exit_data_nowait                | EXIT_DATA_NOWAIT          |
|                           | ompt_target_update_nowait                   | UPDATE_NOWAIT             |
| **ompt_target_data_op_t** |                                             |                           |
|                           | ompt_target_data_alloc                      | ALLOC                     |
|                           | ompt_target_data_transfer_to_device         | H2D                       |
|                           | ompt_target_data_transfer_from_device       | D2H                       |
|                           | ompt_target_data_delete                     | DELETE                    |
|                           | ompt_target_data_associate                  | ASSOCIATE                 |
|                           | ompt_target_data_disassociate               | DISASSOCIATE              |
|                           | ompt_target_data_alloc_async                | ALLOC_ASYNC               |
|                           | ompt_target_data_transfer_to_device_async   | H2D_ASYNC                 |
|                           | ompt_target_data_transfer_from_device_async | D2H_ASYNC                 |
|                           | ompt_target_data_delete_async               | DELETE_ASYNC              |
| **ompt_callbacks_t**      |                                             |                           |
|                           | ompt_callback_target                        | CB_TARGET                 |
|                           | ompt_callback_target_data_op                | CB_DATAOP                 |
|                           | ompt_callback_target_submit                 | CB_KERNEL                 |
| **ompt_work_t**           |                                             |                           |
|                           | ompt_work_loop                              | WORK_LOOP                 |
|                           | ompt_work_sections                          | WORK_SECT                 |
|                           | ompt_work_single_executor                   | WORK_EXEC                 |
|                           | ompt_work_single_other                      | WORK_SINGLE               |
|                           | ompt_work_workshare                         | WORK_SHARE                |
|                           | ompt_work_distribute                        | WORK_DIST                 |
|                           | ompt_work_taskloop                          | WORK_TASK                 |
|                           | ompt_work_scope                             | WORK_SCOPE                |
|                           | ompt_work_loop_static                       | WORK_LOOP_STA             |
|                           | ompt_work_loop_dynamic                      | WORK_LOOP_DYN             |
|                           | ompt_work_loop_guided                       | WORK_LOOP_GUI             |
|                           | ompt_work_loop_other                        | WORK_LOOP_OTH             |
| **ompt_sync_region_t**    |                                             |                           |
|                           | ompt_sync_region_barrier                    | SR_BARRIER                |
|                           | ompt_sync_region_barrier_implicit           | SR_BARRIER_IMPL           |
|                           | ompt_sync_region_barrier_explicit           | SR_BARRIER_EXPL           |
|                           | ompt_sync_region_barrier_implementation     | SR_BARRIER_IMPLEMENTATION |
|                           | ompt_sync_region_taskwait                   | SR_TASKWAIT               |
|                           | ompt_sync_region_taskgroup                  | SR_TASKGROUP              |
|                           | ompt_sync_region_reduction                  | SR_REDUCTION              |
|                           | ompt_sync_region_barrier_implicit_workshare | SR_BARRIER_IMPL_WORKSHARE |
|                           | ompt_sync_region_barrier_implicit_parallel  | SR_BARRIER_IMPL_PARALLEL  |
|                           | ompt_sync_region_barrier_teams              | SR_BARRIER_TEAMS          |


Limitations
===========
Currently, there are some peculiarities which have to be kept in mind when using
this library:

## Callbacks
  * It is not possible to e.g. test non-EMI -AND- EMI callbacks within the same
    test file. Reason: all testsuites share the initialization and therefore the
    registered callbacks.
  * It is not possible to check for device initialization and/or load callbacks
    more than once per test file. The first testcase being run, triggers these
    callbacks and is therefore the only testcase that is able to check for them.
    This is because, after that, the device remains initialized.
  * It is not possible to check for device finalization callbacks, as libomptest
    is un-loaded before this callback occurs. Same holds true for the final
    ThreadEnd event(s).

Miscellaneous
=============

## Default values

To allow for easier writing of tests, many OMPT events may be created using less
parameters than actually requested by the spec -- by using default values. These
defaults are currently set to the corresponding data type's minimum as follows,
for example integers use: `std::numeric_limits<int>::min()`.

When an expected / user-specified event has certain values set to the
corresponding default, these values are ignored. That is, when compared to an
observed event, this property is considered as 'equal' regardless of their
actual equality relation.

References
==========
[0]: ompTest – Unit Testing with OMPT
     https://doi.org/10.1109/SCW63240.2024.00031

[1]: OMPTBench – OpenMP Tool Interface Conformance Testing
     https://doi.org/10.1109/SCW63240.2024.00036
