//===-- OMPDescriptors.inc - OpenMP descriptors ------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#ifdef GEN_OMP_CLAUSE_DESCRIPTORS
  {
    Clause::OMPC_absent,
    descriptor::Clause(
      "absent",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_acq_rel,
    descriptor::Clause(
      "acq_rel",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_acquire,
    descriptor::Clause(
      "acquire",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_adjust_args,
    descriptor::Clause(
      "adjust_args",
      {
        {Version(51), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Repeatable}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp}, {}}},
        {Version(52), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Repeatable}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp}, {}}},
        {Version(60), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Repeatable}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Repeatable}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {ModifierSet::AdjustOp}}},
      }
    ),
  },
  {
    Clause::OMPC_affinity,
    descriptor::Clause(
      "affinity",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TaskInherited}}, "affinity", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TaskInherited}}, "affinity", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_align,
    descriptor::Clause(
      "align",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_aligned,
    descriptor::Clause(
      "aligned",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment, Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_allocate,
    descriptor::Clause(
      "allocate",
      {
        {Version(50), {{{Property::AllPrivatizing, Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "allocate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AllocatorSimpleModifier}, {}}},
        {Version(51), {{{Property::AllPrivatizing, Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "allocate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorSimpleModifier}, {}}},
        {Version(52), {{{Property::AllPrivatizing, Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier}, {}}},
        {Version(60), {{{Property::AllPrivatizing, Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::AllPrivatizing, Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier, Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_allocator,
    descriptor::Clause(
      "allocator",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_append_args,
    descriptor::Clause(
      "append_args",
      {
        {Version(51), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DispatchCompliant, Property::Free, Property::Optional, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_apply,
    descriptor::Clause(
      "apply",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "apply", {Directive::OMPD_fuse, Directive::OMPD_interchange, Directive::OMPD_nothing, Directive::OMPD_reverse, Directive::OMPD_split, Directive::OMPD_stripe, Directive::OMPD_tile, Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LoopModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "apply", {Directive::OMPD_flatten, Directive::OMPD_fuse, Directive::OMPD_interchange, Directive::OMPD_nothing, Directive::OMPD_reverse, Directive::OMPD_split, Directive::OMPD_stripe, Directive::OMPD_tile, Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LoopModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_at,
    descriptor::Clause(
      "at",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_atomic_default_mem_order,
    descriptor::Clause(
      "atomic_default_mem_order",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_bind,
    descriptor::Clause(
      "bind",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_cancellation_construct_type,
    descriptor::Clause(
      "cancellation_construct_type",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "cancellation_construct_type", {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "cancellation_construct_type", {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "cancellation_construct_type", {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "cancellation_construct_type", {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "cancellation_construct_type", {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "cancellation_construct_type", {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_capture,
    descriptor::Clause(
      "capture",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_collapse,
    descriptor::Clause(
      "collapse",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_collector,
    descriptor::Clause(
      "collector",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "collector", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "collector", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_combiner,
    descriptor::Clause(
      "combiner",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "combiner", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "combiner", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_compare,
    descriptor::Clause(
      "compare",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_contains,
    descriptor::Clause(
      "contains",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_copyin,
    descriptor::Clause(
      "copyin",
      {
        {Version(45), {{{Property::Compatible, Property::DataCopying, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataCopying, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataCopying, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataCopying, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataCopying, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataCopying, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_copyprivate,
    descriptor::Clause(
      "copyprivate",
      {
        {Version(45), {{{Property::Compatible, Property::DataCopying, Property::EndClause, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataCopying, Property::EndClause, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataCopying, Property::EndClause, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataCopying, Property::EndClause, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataCopying, Property::EndClause, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataCopying, Property::EndClause, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_counts,
    descriptor::Clause(
      "counts",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "counts", {Directive::OMPD_split}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "counts", {Directive::OMPD_split}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_default,
    descriptor::Clause(
      "default",
      {
        {Version(45), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(51), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(52), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(60), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::VariableCategory}, {}}},
        {Version(61), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "default", {Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {ModifierSet::VariableSelector}}},
      }
    ),
  },
  {
    Clause::OMPC_default_variant,
    descriptor::Clause(
      "default-variant",
      {
        {Version(50), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, "default-variant", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, "default-variant", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_defaultmap,
    descriptor::Clause(
      "defaultmap",
      {
        {Version(45), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(50), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(51), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(52), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}, {}}},
        {Version(60), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::VariableCategory}, {}}},
        {Version(61), {{{Property::Compatible, Property::DefaultAttribute, Property::Free, Property::Optional, Property::PostModified, Property::Repeatable}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {ModifierSet::VariableSelector}}},
      }
    ),
  },
  {
    Clause::OMPC_depend,
    descriptor::Clause(
      "depend",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "depend", {Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::TaskDependenceType}, {ModifierSet::DependModifierSet1}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::Iterator, Modifier::TaskDependenceType}, {ModifierSet::DependModifierSet1}}},
        {Version(51), {{{Property::Compatible, Property::DispatchRequirement, Property::Free, Property::Optional, Property::Repeatable}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::Iterator, Modifier::TaskDependenceType}, {ModifierSet::DependModifierSet1}}},
        {Version(52), {{{Property::Compatible, Property::DispatchRequirement, Property::Free, Property::Optional, Property::Repeatable}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::TaskDependenceType}, {}}},
        {Version(60), {{{Property::Compatible, Property::DispatchRequirement, Property::Free, Property::Optional, Property::Repeatable, Property::TaskInherited, Property::TaskgraphAltering}}, "depend", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::TaskDependenceType}, {}}},
        {Version(61), {{{Property::Compatible, Property::DispatchRequirement, Property::Free, Property::Optional, Property::Repeatable, Property::TaskInherited, Property::TaskSynchronizing, Property::TaskgraphAltering}}, "depend", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::TaskDependenceType}, {ModifierSet::DependListType, ModifierSet::DependModifierSet1}}},
      }
    ),
  },
  {
    Clause::OMPC_depth,
    descriptor::Clause(
      "depth",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "depth", {Directive::OMPD_flatten, Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_destroy,
    descriptor::Clause(
      "destroy",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "destroy", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_detach,
    descriptor::Clause(
      "detach",
      {
        {Version(50), {{{Property::Compatible, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_device,
    descriptor::Clause(
      "device",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "device", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "device", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::IcvDefaulted, Property::Optional, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier, Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_device_safesync,
    descriptor::Clause(
      "device_safesync",
      {
        {Version(60), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "device_safesync", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "device_safesync", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_device_type,
    descriptor::Clause(
      "device_type",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "device_type", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target, Directive::OMPD_groupprivate, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target, Directive::OMPD_groupprivate, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_dist_schedule,
    descriptor::Clause(
      "dist_schedule",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_doacross,
    descriptor::Clause(
      "doacross",
      {
        {Version(52), {{{Property::Compatible, Property::Free, Property::Repeatable, Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Repeatable, Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Repeatable, Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_dyn_groupprivate,
    descriptor::Clause(
      "dyn_groupprivate",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "dyn_groupprivate", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AccessGroup, Modifier::DirectiveNameModifier, Modifier::FallbackModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_dynamic_allocators,
    descriptor::Clause(
      "dynamic_allocators",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_enter,
    descriptor::Clause(
      "enter",
      {
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AutomapModifier, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AutomapModifier, Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_exclusive,
    descriptor::Clause(
      "exclusive",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_fail,
    descriptor::Clause(
      "fail",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_filter,
    descriptor::Clause(
      "filter",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_final,
    descriptor::Clause(
      "final",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_firstprivate,
    descriptor::Clause(
      "firstprivate",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Saved}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Saved}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_from,
    descriptor::Clause(
      "from",
      {
        {Version(45), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Mapper}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::Mapper, Modifier::MotionModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Expectation, Modifier::Iterator, Modifier::Mapper}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_full,
    descriptor::Clause(
      "full",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_grainsize,
    descriptor::Clause(
      "grainsize",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::TaskgraphAltering, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::TaskgraphAltering, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_graph_id,
    descriptor::Clause(
      "graph_id",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "graph_id", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "graph_id", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_graph_reset,
    descriptor::Clause(
      "graph_reset",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "graph_reset", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "graph_reset", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_has_device_addr,
    descriptor::Clause(
      "has_device_addr",
      {
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "has_device_addr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "has_device_addr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "has_device_addr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Repeatable}}, "has_device_addr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_hint,
    descriptor::Clause(
      "hint",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "hint", {Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_holds,
    descriptor::Clause(
      "holds",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_if,
    descriptor::Clause(
      "if",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable, Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_in_reduction,
    descriptor::Clause(
      "in_reduction",
      {
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::ReductionParticipating, Property::Repeatable}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::ReductionParticipating, Property::Repeatable}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::ReductionParticipating, Property::Repeatable}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::ReductionParticipating, Property::Repeatable}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Privatization, Property::ReductionParticipating, Property::Repeatable}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_inbranch,
    descriptor::Clause(
      "inbranch",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_inclusive,
    descriptor::Clause(
      "inclusive",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_indirect,
    descriptor::Clause(
      "indirect",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_induction,
    descriptor::Clause(
      "induction",
      {
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "induction", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::InductionIdentifier, Modifier::InductionModifier, Modifier::StepModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "induction", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::InductionIdentifier, Modifier::InductionModifier, Modifier::StepModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_inductor,
    descriptor::Clause(
      "inductor",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "inductor", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "inductor", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_init,
    descriptor::Clause(
      "init",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "init", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::InteropType}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "init", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::InteropType}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "init", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DepinfoModifier, Modifier::DirectiveNameModifier, Modifier::InteropType, Modifier::PreferType}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Repeatable}}, "init", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DepinfoModifier, Modifier::DirectiveNameModifier, Modifier::PreferType}, {ModifierSet::InteropType}}},
      }
    ),
  },
  {
    Clause::OMPC_init_complete,
    descriptor::Clause(
      "init_complete",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "init_complete", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "init_complete", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_initializer,
    descriptor::Clause(
      "initializer",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_interop,
    descriptor::Clause(
      "interop",
      {
        {Version(60), {{{Property::Compatible, Property::DispatchRequirement, Property::Free, Property::Optional, Property::Unique}}, "interop", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DispatchRequirement, Property::Free, Property::Optional, Property::Unique}}, "interop", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_is_device_ptr,
    descriptor::Clause(
      "is_device_ptr",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "is_device_ptr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "is_device_ptr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_lastprivate,
    descriptor::Clause(
      "lastprivate",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LastprivateModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::Repeatable}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LastprivateModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_linear,
    descriptor::Clause(
      "linear",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::PostModified, Property::Privatization, Property::Repeatable}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::PostModified, Property::Privatization, Property::Repeatable}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::PostModified, Property::Privatization, Property::Repeatable}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::PostModified, Property::Privatization, Property::Repeatable}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::PostModified, Property::Privatization, Property::Repeatable}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::PostModified, Property::Privatization, Property::Repeatable}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_link,
    descriptor::Clause(
      "link",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_local,
    descriptor::Clause(
      "local",
      {
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "local", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "local", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_looprange,
    descriptor::Clause(
      "looprange",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "looprange", {Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "looprange", {Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_map,
    descriptor::Clause(
      "map",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "map", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MapType, Modifier::MapTypeModifier, Modifier::OmpxHoldModifier}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlwaysModifier, Modifier::CloseModifier, Modifier::DeleteModifier, Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::MapType, Modifier::Mapper, Modifier::OmpxHoldModifier, Modifier::PresentModifier, Modifier::RefModifier, Modifier::SelfModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataMappingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AttachModifier, Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::MapType, Modifier::Mapper, Modifier::OmpxHoldModifier, Modifier::RefModifier}, {ModifierSet::MapTypeModifying}}},
      }
    ),
  },
  {
    Clause::OMPC_match,
    descriptor::Clause(
      "match",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "match", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_memscope,
    descriptor::Clause(
      "memscope",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "memscope", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "memscope", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_mergeable,
    descriptor::Clause(
      "mergeable",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "mergeable", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "mergeable", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_message,
    descriptor::Clause(
      "message",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "message", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "message", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "message", {Directive::OMPD_error, Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "message", {Directive::OMPD_error, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp,
    descriptor::Clause(
      "no_openmp",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp_constructs,
    descriptor::Clause(
      "no_openmp_constructs",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp_constructs", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp_constructs", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp_routines,
    descriptor::Clause(
      "no_openmp_routines",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_no_parallelism,
    descriptor::Clause(
      "no_parallelism",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_nocontext,
    descriptor::Clause(
      "nocontext",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_nogroup,
    descriptor::Clause(
      "nogroup",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_target_data, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::OutermostLeaf, Property::TaskSynchronizing, Property::Unique}}, "nogroup", {Directive::OMPD_target_data, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_nontemporal,
    descriptor::Clause(
      "nontemporal",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_notinbranch,
    descriptor::Clause(
      "notinbranch",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_novariants,
    descriptor::Clause(
      "novariants",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_nowait,
    descriptor::Clause(
      "nowait",
      {
        {Version(45), {{{Property::Compatible, Property::EndClause, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::EndClause, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DispatchRequirement, Property::EndClause, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DispatchRequirement, Property::EndClause, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DispatchRequirement, Property::EndClause, Property::Free, Property::Optional, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DispatchRequirement, Property::EndClause, Property::Free, Property::Optional, Property::OutermostLeaf, Property::TaskSynchronizing, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_num_tasks,
    descriptor::Clause(
      "num_tasks",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::TaskgraphAltering, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::TaskgraphAltering, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_num_teams,
    descriptor::Clause(
      "num_teams",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LowerBound}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::SpaceConfiguring, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::LowerBound}, {ModifierSet::NumTeamsModifierSet1}}},
      }
    ),
  },
  {
    Clause::OMPC_num_threads,
    descriptor::Clause(
      "num_threads",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::SpaceConfiguring, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::PrescriptivenessModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_ompx_bare,
    descriptor::Clause(
      "ompx_bare",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "ompx_bare", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "ompx_bare", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "ompx_bare", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "ompx_bare", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "ompx_bare", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "ompx_bare", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_order,
    descriptor::Clause(
      "order",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::OrderModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::OrderModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OrderModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OrderModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_ordered,
    descriptor::Clause(
      "ordered",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::OnceForAllConstituents, Property::Optional, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_otherwise,
    descriptor::Clause(
      "otherwise",
      {
        {Version(52), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_partial,
    descriptor::Clause(
      "partial",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_permutation,
    descriptor::Clause(
      "permutation",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "permutation", {Directive::OMPD_interchange}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "permutation", {Directive::OMPD_interchange}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_priority,
    descriptor::Clause(
      "priority",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "priority", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "priority", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_private,
    descriptor::Clause(
      "private",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Privatization, Property::Repeatable}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_proc_bind,
    descriptor::Clause(
      "proc_bind",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_read,
    descriptor::Clause(
      "read",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_reduction,
    descriptor::Clause(
      "reduction",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping, Property::Repeatable}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping, Property::Repeatable}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping, Property::Repeatable}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping, Property::Repeatable}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping, Property::Repeatable}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OriginalSharingModifier, Modifier::ReductionIdentifier, Modifier::ReductionModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping, Property::Repeatable}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OriginalSharingModifier, Modifier::PerThreadModifier, Modifier::ReductionIdentifier}, {ModifierSet::ReductionModifierSet1, ModifierSet::ReductionType}}},
      }
    ),
  },
  {
    Clause::OMPC_reduction_kind,
    descriptor::Clause(
      "reduction_kind",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "reduction_kind", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_relaxed,
    descriptor::Clause(
      "relaxed",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_release,
    descriptor::Clause(
      "release",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_replayable,
    descriptor::Clause(
      "replayable",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "replayable", {Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "replayable", {Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_reverse_offload,
    descriptor::Clause(
      "reverse_offload",
      {
        {Version(50), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_safelen,
    descriptor::Clause(
      "safelen",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_safesync,
    descriptor::Clause(
      "safesync",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safesync", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "safesync", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_schedule,
    descriptor::Clause(
      "schedule",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::DirectiveNameModifier, Modifier::OrderingModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::DirectiveNameModifier, Modifier::OrderingModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_self_maps,
    descriptor::Clause(
      "self_maps",
      {
        {Version(60), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "self_maps", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "self_maps", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_seq_cst,
    descriptor::Clause(
      "seq_cst",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "seq_cst", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_severity,
    descriptor::Clause(
      "severity",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "severity", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "severity", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "severity", {Directive::OMPD_error, Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "severity", {Directive::OMPD_error, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_shared,
    descriptor::Clause(
      "shared",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_shift,
    descriptor::Clause(
      "shift",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "shift", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_simd,
    descriptor::Clause(
      "simd",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_simdlen,
    descriptor::Clause(
      "simdlen",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ScaledModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_sizes,
    descriptor::Clause(
      "sizes",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_stripe, Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_stripe, Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::SizesSelector}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_task_reduction,
    descriptor::Clause(
      "task_reduction",
      {
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping, Property::Repeatable}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping, Property::Repeatable}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping, Property::Repeatable}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping, Property::Repeatable}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping, Property::Repeatable}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_thread_limit,
    descriptor::Clause(
      "thread_limit",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::IcvModifying, Property::Optional, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::IcvModifying, Property::Optional, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::IcvModifying, Property::Optional, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::IcvModifying, Property::Optional, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::IcvModifying, Property::Optional, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::IcvModifying, Property::Optional, Property::SpaceConfiguring, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::PrescriptivenessModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_threads,
    descriptor::Clause(
      "threads",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_threadset,
    descriptor::Clause(
      "threadset",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "threadset", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "threadset", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_to,
    descriptor::Clause(
      "to",
      {
        {Version(45), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Mapper}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::Mapper, Modifier::MotionModifier}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Expectation, Modifier::Iterator, Modifier::Mapper}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataMotionAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_transparent,
    descriptor::Clause(
      "transparent",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "transparent", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "transparent", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_unified_address,
    descriptor::Clause(
      "unified_address",
      {
        {Version(50), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_unified_shared_memory,
    descriptor::Clause(
      "unified_shared_memory",
      {
        {Version(50), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DeviceGlobalRequirement, Property::Free, Property::Optional, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_uniform,
    descriptor::Clause(
      "uniform",
      {
        {Version(45), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_untied,
    descriptor::Clause(
      "untied",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_update,
    descriptor::Clause(
      "update",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_update_depend_objects,
    descriptor::Clause(
      "update-depend_objects",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::TaskDependenceType}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::TaskDependenceType}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_use,
    descriptor::Clause(
      "use",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_use_device_addr,
    descriptor::Clause(
      "use_device_addr",
      {
        {Version(50), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Repeatable}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Repeatable}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Repeatable}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Repeatable}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Repeatable}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_use_device_ptr,
    descriptor::Clause(
      "use_device_ptr",
      {
        {Version(45), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::AllDataEnvironments, Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Free, Property::Optional, Property::Privatization, Property::Repeatable}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Fallback}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_uses_allocators,
    descriptor::Clause(
      "uses_allocators",
      {
        {Version(50), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MemSpace, Modifier::TraitsArray}, {}}},
        {Version(60), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::MemSpace, Modifier::TraitsArray}, {}}},
        {Version(61), {{{Property::Compatible, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Free, Property::Optional, Property::Repeatable}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::MemSpace, Modifier::TraitsArray}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_weak,
    descriptor::Clause(
      "weak",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_when,
    descriptor::Clause(
      "when",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector, Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector, Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
  {
    Clause::OMPC_write,
    descriptor::Clause(
      "write",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}, {}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::InnermostLeaf, Property::Optional, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}, {}}},
      }
    ),
  },
#endif // GEN_OMP_CLAUSE_DESCRIPTORS
#ifdef GEN_OMP_CLAUSE_GROUP_DESCRIPTORS
  {
    ClauseSet::Assumption,
    descriptor::ClauseSet(
      "assumption",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_contains, Clause::OMPC_holds, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism}, {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_contains, Clause::OMPC_holds, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism}, {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_contains, Clause::OMPC_holds, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism}, {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_contains, Clause::OMPC_holds, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism}, {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}}},
      }
    ),
  },
  {
    ClauseSet::Atomic,
    descriptor::ClauseSet(
      "atomic",
      {
        {Version(45), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_read, Clause::OMPC_update, Clause::OMPC_write}, {Directive::OMPD_atomic}}},
        {Version(50), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_read, Clause::OMPC_update, Clause::OMPC_write}, {Directive::OMPD_atomic}}},
        {Version(51), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_read, Clause::OMPC_update, Clause::OMPC_write}, {Directive::OMPD_atomic}}},
        {Version(52), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_read, Clause::OMPC_update, Clause::OMPC_write}, {Directive::OMPD_atomic}}},
        {Version(60), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_read, Clause::OMPC_update, Clause::OMPC_write}, {Directive::OMPD_atomic}}},
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_read, Clause::OMPC_update, Clause::OMPC_write}, {Directive::OMPD_atomic}}},
      }
    ),
  },
  {
    ClauseSet::Branch,
    descriptor::ClauseSet(
      "branch",
      {
        {Version(45), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_inbranch, Clause::OMPC_notinbranch}, {Directive::OMPD_declare_simd}}},
        {Version(50), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_inbranch, Clause::OMPC_notinbranch}, {Directive::OMPD_declare_simd}}},
        {Version(51), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_inbranch, Clause::OMPC_notinbranch}, {Directive::OMPD_declare_simd}}},
        {Version(52), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_inbranch, Clause::OMPC_notinbranch}, {Directive::OMPD_declare_simd}}},
        {Version(60), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_inbranch, Clause::OMPC_notinbranch}, {Directive::OMPD_declare_simd}}},
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_inbranch, Clause::OMPC_notinbranch}, {Directive::OMPD_declare_simd}}},
      }
    ),
  },
  {
    ClauseSet::CancelDirectiveName,
    descriptor::ClauseSet(
      "cancel-directive-name",
      {
        {Version(50), {{{Property::Exclusive, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_cancellation_construct_type}, {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}}},
        {Version(51), {{{Property::Exclusive, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_cancellation_construct_type}, {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}}},
        {Version(52), {{{Property::Exclusive, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_cancellation_construct_type}, {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}}},
        {Version(60), {{{Property::Exclusive, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_cancellation_construct_type}, {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}}},
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_cancellation_construct_type}, {Directive::OMPD_cancel, Directive::OMPD_cancellation_point}}},
      }
    ),
  },
  {
    ClauseSet::ExtendedAtomic,
    descriptor::ClauseSet(
      "extended-atomic",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_capture}, {Directive::OMPD_atomic}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_capture}, {Directive::OMPD_atomic}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_capture, Clause::OMPC_compare, Clause::OMPC_fail, Clause::OMPC_weak}, {Directive::OMPD_atomic}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_capture, Clause::OMPC_compare, Clause::OMPC_fail, Clause::OMPC_weak}, {Directive::OMPD_atomic}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_capture, Clause::OMPC_compare, Clause::OMPC_fail, Clause::OMPC_weak}, {Directive::OMPD_atomic}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_capture, Clause::OMPC_compare, Clause::OMPC_fail, Clause::OMPC_weak}, {Directive::OMPD_atomic}}},
      }
    ),
  },
  {
    ClauseSet::MemoryOrder,
    descriptor::ClauseSet(
      "memory-order",
      {
        {Version(45), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_seq_cst}, {Directive::OMPD_atomic}}},
        {Version(50), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_seq_cst}, {Directive::OMPD_atomic, Directive::OMPD_flush}}},
        {Version(51), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_seq_cst}, {Directive::OMPD_atomic, Directive::OMPD_flush}}},
        {Version(52), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_seq_cst}, {Directive::OMPD_atomic, Directive::OMPD_flush}}},
        {Version(60), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_seq_cst}, {Directive::OMPD_atomic, Directive::OMPD_flush}}},
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_seq_cst}, {Directive::OMPD_atomic, Directive::OMPD_flush}}},
      }
    ),
  },
  {
    ClauseSet::ParallelizationLevel,
    descriptor::ClauseSet(
      "parallelization-level",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_simd, Clause::OMPC_threads}, {Directive::OMPD_ordered_blockassoc}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_simd, Clause::OMPC_threads}, {Directive::OMPD_ordered_blockassoc}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_simd, Clause::OMPC_threads}, {Directive::OMPD_ordered_blockassoc}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_simd, Clause::OMPC_threads}, {Directive::OMPD_ordered_blockassoc}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_simd, Clause::OMPC_threads}, {Directive::OMPD_ordered_blockassoc}}},
      }
    ),
  },
  {
    ClauseSet::Requirement,
    descriptor::ClauseSet(
      "requirement",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_atomic_default_mem_order, Clause::OMPC_dynamic_allocators, Clause::OMPC_reverse_offload, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory}, {Directive::OMPD_requires}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_atomic_default_mem_order, Clause::OMPC_dynamic_allocators, Clause::OMPC_reverse_offload, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory}, {Directive::OMPD_requires}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_atomic_default_mem_order, Clause::OMPC_dynamic_allocators, Clause::OMPC_reverse_offload, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory}, {Directive::OMPD_requires}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_atomic_default_mem_order, Clause::OMPC_device_safesync, Clause::OMPC_dynamic_allocators, Clause::OMPC_reverse_offload, Clause::OMPC_self_maps, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory}, {Directive::OMPD_requires}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_atomic_default_mem_order, Clause::OMPC_device_safesync, Clause::OMPC_dynamic_allocators, Clause::OMPC_reverse_offload, Clause::OMPC_self_maps, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory}, {Directive::OMPD_requires}}},
      }
    ),
  },
#endif // GEN_OMP_CLAUSE_GROUP_DESCRIPTORS
#ifdef GEN_OMP_CLAUSE_SET_DESCRIPTORS
  {
    ClauseSet::ActionClause,
    descriptor::ClauseSet(
      "action-clause",
      {
        {Version(51), {{{Property::Required}}, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_use}, {Directive::OMPD_interop}}},
        {Version(52), {{{Property::Required}}, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_use}, {Directive::OMPD_interop}}},
        {Version(60), {{{Property::Required}}, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_use}, {Directive::OMPD_interop}}},
        {Version(61), {{{Property::Required}}, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_use}, {Directive::OMPD_interop}}},
      }
    ),
  },
  {
    ClauseSet::DataEnvironmentClause,
    descriptor::ClauseSet(
      "data-environment-clause",
      {
        {Version(50), {{{Property::Required}}, {Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {Directive::OMPD_target_data}}},
        {Version(51), {{{Property::Required}}, {Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {Directive::OMPD_target_data}}},
        {Version(52), {{{Property::Required}}, {Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {Directive::OMPD_target_data}}},
        {Version(60), {{{Property::Required}}, {Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {Directive::OMPD_target_data}}},
        {Version(61), {{{Property::Required}}, {Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {Directive::OMPD_target_data}}},
      }
    ),
  },
  {
    ClauseSet::DepobjClauseSet1,
    descriptor::ClauseSet(
      "depobj_clause_set_1",
      {
        {Version(50), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_update_depend_objects}, {Directive::OMPD_depobj}}},
        {Version(51), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_update_depend_objects}, {Directive::OMPD_depobj}}},
        {Version(52), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_update_depend_objects}, {Directive::OMPD_depobj}}},
        {Version(60), {{{Property::Required}}, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_update_depend_objects}, {Directive::OMPD_depobj}}},
        {Version(61), {{{Property::Required}}, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_update_depend_objects}, {Directive::OMPD_depobj}}},
      }
    ),
  },
  {
    ClauseSet::GranularityClause,
    descriptor::ClauseSet(
      "granularity-clause",
      {
        {Version(45), {{{Property::Exclusive}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}, {Directive::OMPD_taskloop}}},
        {Version(50), {{{Property::Exclusive}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}, {Directive::OMPD_taskloop}}},
        {Version(51), {{{Property::Exclusive}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}, {Directive::OMPD_taskloop}}},
        {Version(52), {{{Property::Exclusive}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}, {Directive::OMPD_taskloop}}},
        {Version(60), {{{Property::Exclusive}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}, {Directive::OMPD_taskloop}}},
        {Version(61), {{{Property::Exclusive}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}, {Directive::OMPD_taskloop}}},
      }
    ),
  },
  {
    ClauseSet::ScanClauseSet1,
    descriptor::ClauseSet(
      "scan_clause_set_1",
      {
        {Version(50), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_exclusive, Clause::OMPC_inclusive}, {Directive::OMPD_scan}}},
        {Version(51), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_exclusive, Clause::OMPC_inclusive}, {Directive::OMPD_scan}}},
        {Version(52), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_exclusive, Clause::OMPC_inclusive}, {Directive::OMPD_scan}}},
        {Version(60), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_exclusive, Clause::OMPC_inclusive, Clause::OMPC_init_complete}, {Directive::OMPD_scan}}},
        {Version(61), {{{Property::Exclusive, Property::Required, Property::Unique}}, {Clause::OMPC_exclusive, Clause::OMPC_inclusive, Clause::OMPC_init_complete}, {Directive::OMPD_scan}}},
      }
    ),
  },
  {
    ClauseSet::SingleClauseSet1,
    descriptor::ClauseSet(
      "single_clause_set_1",
      {
        {Version(45), {{{Property::Exclusive}}, {Clause::OMPC_copyprivate, Clause::OMPC_nowait}, {Directive::OMPD_single}}},
        {Version(50), {{{Property::Exclusive}}, {Clause::OMPC_copyprivate, Clause::OMPC_nowait}, {Directive::OMPD_single}}},
        {Version(51), {{{Property::Exclusive}}, {Clause::OMPC_copyprivate, Clause::OMPC_nowait}, {Directive::OMPD_single}}},
        {Version(52), {{{Property::Exclusive}}, {Clause::OMPC_copyprivate, Clause::OMPC_nowait}, {Directive::OMPD_single}}},
        {Version(60), {{{Property::Exclusive}}, {Clause::OMPC_copyprivate, Clause::OMPC_nowait}, {Directive::OMPD_single}}},
        {Version(61), {{{Property::Exclusive}}, {Clause::OMPC_copyprivate, Clause::OMPC_nowait}, {Directive::OMPD_single}}},
      }
    ),
  },
  {
    ClauseSet::SynchronizationClause,
    descriptor::ClauseSet(
      "synchronization-clause",
      {
        {Version(50), {{{Property::Exclusive}}, {Clause::OMPC_nogroup, Clause::OMPC_reduction}, {Directive::OMPD_taskloop}}},
        {Version(51), {{{Property::Exclusive}}, {Clause::OMPC_nogroup, Clause::OMPC_reduction}, {Directive::OMPD_taskloop}}},
        {Version(52), {{{Property::Exclusive}}, {Clause::OMPC_nogroup, Clause::OMPC_reduction}, {Directive::OMPD_taskloop}}},
        {Version(60), {{{Property::Exclusive}}, {Clause::OMPC_nogroup, Clause::OMPC_reduction}, {Directive::OMPD_taskloop}}},
        {Version(61), {{{Property::Exclusive}}, {Clause::OMPC_nogroup, Clause::OMPC_reduction}, {Directive::OMPD_taskloop}}},
      }
    ),
  },
  {
    ClauseSet::TargetDataClauseSet1,
    descriptor::ClauseSet(
      "target_data_clause_set_1",
      {
        {Version(45), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_data}}},
      }
    ),
  },
  {
    ClauseSet::TargetEnterDataClauseSet1,
    descriptor::ClauseSet(
      "target_enter_data_clause_set_1",
      {
        {Version(45), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_enter_data}}},
        {Version(50), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_enter_data}}},
        {Version(51), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_enter_data}}},
        {Version(52), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_enter_data}}},
        {Version(60), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_enter_data}}},
        {Version(61), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_enter_data}}},
      }
    ),
  },
  {
    ClauseSet::TargetExitDataClauseSet1,
    descriptor::ClauseSet(
      "target_exit_data_clause_set_1",
      {
        {Version(45), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_exit_data}}},
        {Version(50), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_exit_data}}},
        {Version(51), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_exit_data}}},
        {Version(52), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_exit_data}}},
        {Version(60), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_exit_data}}},
        {Version(61), {{{Property::Required}}, {Clause::OMPC_map}, {Directive::OMPD_target_exit_data}}},
      }
    ),
  },
  {
    ClauseSet::TargetUpdateClauseSet1,
    descriptor::ClauseSet(
      "target_update_clause_set_1",
      {
        {Version(45), {{{Property::Required}}, {Clause::OMPC_from, Clause::OMPC_to}, {Directive::OMPD_target_update}}},
        {Version(50), {{{Property::Required}}, {Clause::OMPC_from, Clause::OMPC_to}, {Directive::OMPD_target_update}}},
        {Version(51), {{{Property::Required}}, {Clause::OMPC_from, Clause::OMPC_to}, {Directive::OMPD_target_update}}},
        {Version(52), {{{Property::Required}}, {Clause::OMPC_from, Clause::OMPC_to}, {Directive::OMPD_target_update}}},
        {Version(60), {{{Property::Required}}, {Clause::OMPC_from, Clause::OMPC_to}, {Directive::OMPD_target_update}}},
        {Version(61), {{{Property::Required}}, {Clause::OMPC_from, Clause::OMPC_to}, {Directive::OMPD_target_update}}},
      }
    ),
  },
  {
    ClauseSet::TaskClauseSet1,
    descriptor::ClauseSet(
      "task_clause_set_1",
      {
        {Version(50), {{{Property::Exclusive}}, {Clause::OMPC_detach, Clause::OMPC_mergeable}, {Directive::OMPD_task}}},
        {Version(51), {{{Property::Exclusive}}, {Clause::OMPC_detach, Clause::OMPC_mergeable}, {Directive::OMPD_task}}},
        {Version(52), {{{Property::Exclusive}}, {Clause::OMPC_detach, Clause::OMPC_mergeable}, {Directive::OMPD_task}}},
        {Version(60), {{{Property::Exclusive}}, {Clause::OMPC_detach, Clause::OMPC_mergeable}, {Directive::OMPD_task}}},
        {Version(61), {{{Property::Exclusive}}, {Clause::OMPC_detach, Clause::OMPC_mergeable}, {Directive::OMPD_task}}},
      }
    ),
  },
  {
    ClauseSet::UnrollClauseSet1,
    descriptor::ClauseSet(
      "unroll_clause_set_1",
      {
        {Version(51), {{{Property::Exclusive}}, {Clause::OMPC_full, Clause::OMPC_partial}, {Directive::OMPD_unroll}}},
        {Version(52), {{{Property::Exclusive}}, {Clause::OMPC_full, Clause::OMPC_partial}, {Directive::OMPD_unroll}}},
        {Version(60), {{{Property::Exclusive}}, {Clause::OMPC_full, Clause::OMPC_partial}, {Directive::OMPD_unroll}}},
        {Version(61), {{{Property::Exclusive}}, {Clause::OMPC_full, Clause::OMPC_partial}, {Directive::OMPD_unroll}}},
      }
    ),
  },
#endif // GEN_OMP_CLAUSE_SET_DESCRIPTORS
#ifdef GEN_OMP_DIRECTIVE_DESCRIPTORS
  {
    Directive::OMPD_allocate,
    descriptor::Directive(
      "allocate",
      {
        {Version(50), {{{Property::Pure}}, "allocate", Association::Explicit, Category::Declarative, {Clause::OMPC_allocator}, {}}},
        {Version(51), {{{Property::Pure}}, "allocate", Association::Explicit, Category::Declarative, {Clause::OMPC_align, Clause::OMPC_allocator}, {}}},
        {Version(52), {{{Property::Pure}}, "allocate", Association::Explicit, Category::Declarative, {Clause::OMPC_align, Clause::OMPC_allocator}, {}}},
        {Version(60), {{{Property::Pure}}, "allocate", Association::Explicit, Category::Declarative, {Clause::OMPC_align, Clause::OMPC_allocator}, {}}},
        {Version(61), {{{Property::Pure}}, "allocate", Association::Explicit, Category::Declarative, {Clause::OMPC_align, Clause::OMPC_allocator}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_allocators,
    descriptor::Directive(
      "allocators",
      {
        {Version(52), {{{}}, "allocators", Association::Block, Category::Executable, {Clause::OMPC_allocate}, {}}},
        {Version(60), {{{}}, "allocators", Association::Block, Category::Executable, {Clause::OMPC_allocate}, {}}},
        {Version(61), {{{}}, "allocators", Association::Block, Category::Executable, {Clause::OMPC_allocate}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_assume,
    descriptor::Directive(
      "assume",
      {
        {Version(51), {{{Property::Pure}}, "assume", Association::Block, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(52), {{{Property::Pure}}, "assume", Association::Block, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(60), {{{Property::Pure}}, "assume", Association::Block, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(61), {{{Property::Pure}}, "assume", Association::Block, Category::Informational, {}, {ClauseSet::Assumption}}},
      }
    ),
  },
  {
    Directive::OMPD_assumes,
    descriptor::Directive(
      "assumes",
      {
        {Version(51), {{{Property::Pure}}, "assumes", Association::Unassociated, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(52), {{{Property::Pure}}, "assumes", Association::Unassociated, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(60), {{{Property::Pure}}, "assumes", Association::Unassociated, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(61), {{{Property::Pure}}, "assumes", Association::Unassociated, Category::Informational, {}, {ClauseSet::Assumption}}},
      }
    ),
  },
  {
    Directive::OMPD_atomic,
    descriptor::Directive(
      "atomic",
      {
        {Version(45), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitStrongFlush, Property::ImplicitStrongFlushAtEntry, Property::ImplicitStrongFlushAtExit, Property::MutualExclusion, Property::OrderConcurrentNestable, Property::Simdizable}}, "atomic", Association::Block, Category::Executable, {}, {ClauseSet::Atomic, ClauseSet::ExtendedAtomic, ClauseSet::MemoryOrder}}},
        {Version(50), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitStrongFlush, Property::ImplicitStrongFlushAtEntry, Property::ImplicitStrongFlushAtExit, Property::MutualExclusion, Property::OrderConcurrentNestable, Property::Simdizable}}, "atomic", Association::Block, Category::Executable, {Clause::OMPC_hint}, {ClauseSet::Atomic, ClauseSet::ExtendedAtomic, ClauseSet::MemoryOrder}}},
        {Version(51), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitStrongFlush, Property::ImplicitStrongFlushAtEntry, Property::ImplicitStrongFlushAtExit, Property::MutualExclusion, Property::OrderConcurrentNestable, Property::Simdizable}}, "atomic", Association::Block, Category::Executable, {Clause::OMPC_hint}, {ClauseSet::Atomic, ClauseSet::ExtendedAtomic, ClauseSet::MemoryOrder}}},
        {Version(52), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitStrongFlush, Property::ImplicitStrongFlushAtEntry, Property::ImplicitStrongFlushAtExit, Property::MutualExclusion, Property::OrderConcurrentNestable, Property::Simdizable}}, "atomic", Association::Block, Category::Executable, {Clause::OMPC_hint}, {ClauseSet::Atomic, ClauseSet::ExtendedAtomic, ClauseSet::MemoryOrder}}},
        {Version(60), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitStrongFlush, Property::ImplicitStrongFlushAtEntry, Property::ImplicitStrongFlushAtExit, Property::MutualExclusion, Property::OrderConcurrentNestable, Property::Simdizable}}, "atomic", Association::Block, Category::Executable, {Clause::OMPC_hint, Clause::OMPC_memscope}, {ClauseSet::Atomic, ClauseSet::ExtendedAtomic, ClauseSet::MemoryOrder}}},
        {Version(61), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitStrongFlush, Property::ImplicitStrongFlushAtEntry, Property::ImplicitStrongFlushAtExit, Property::MutualExclusion, Property::OrderConcurrentNestable, Property::Simdizable}}, "atomic", Association::Block, Category::Executable, {Clause::OMPC_hint, Clause::OMPC_memscope}, {ClauseSet::Atomic, ClauseSet::ExtendedAtomic, ClauseSet::MemoryOrder}}},
      }
    ),
  },
  {
    Directive::OMPD_barrier,
    descriptor::Directive(
      "barrier",
      {
        {Version(45), {{{Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TeamExecuted}}, "barrier", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(50), {{{Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TeamExecuted}}, "barrier", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(51), {{{Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TeamExecuted}}, "barrier", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(52), {{{Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TeamExecuted}}, "barrier", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(60), {{{Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TeamExecuted}}, "barrier", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(61), {{{Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TaskSynchronizing, Property::TeamExecuted}}, "barrier", Association::Unassociated, Category::Executable, {}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_begin_assumes,
    descriptor::Directive(
      "begin_assumes",
      {
        {Version(51), {{{}}, "begin assumes", Association::Delimited, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(52), {{{}}, "begin assumes", Association::Delimited, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(60), {{{}}, "begin assumes", Association::Delimited, Category::Informational, {}, {ClauseSet::Assumption}}},
        {Version(61), {{{}}, "begin assumes", Association::Delimited, Category::Informational, {}, {ClauseSet::Assumption}}},
      }
    ),
  },
  {
    Directive::OMPD_begin_declare_target,
    descriptor::Directive(
      "begin_declare_target",
      {
        {Version(51), {{{Property::DeclareTarget, Property::Device, Property::VariantGenerating}}, "begin declare target", Association::Delimited, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_indirect}, {}}},
        {Version(52), {{{Property::DeclareTarget, Property::Device, Property::VariantGenerating}}, "begin declare target", Association::Delimited, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_indirect}, {}}},
        {Version(60), {{{Property::DeclareTarget, Property::Device, Property::VariantGenerating}}, "begin declare_target", Association::Delimited, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_indirect}, {}}},
        {Version(61), {{{Property::DeclareTarget, Property::Device, Property::VariantGenerating}}, "begin declare_target", Association::Delimited, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_indirect}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_begin_declare_variant,
    descriptor::Directive(
      "begin_declare_variant",
      {
        {Version(51), {{{}}, "begin declare variant", Association::Delimited, Category::Declarative, {Clause::OMPC_match}, {}}},
        {Version(52), {{{}}, "begin declare variant", Association::Delimited, Category::Declarative, {Clause::OMPC_match}, {}}},
        {Version(60), {{{}}, "begin declare_variant", Association::Delimited, Category::Declarative, {Clause::OMPC_match}, {}}},
        {Version(61), {{{}}, "begin declare_variant", Association::Delimited, Category::Declarative, {Clause::OMPC_match}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_cancel,
    descriptor::Directive(
      "cancel",
      {
        {Version(45), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancel", Association::Unassociated, Category::Executable, {Clause::OMPC_cancellation_construct_type, Clause::OMPC_if}, {}}},
        {Version(50), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancel", Association::Unassociated, Category::Executable, {Clause::OMPC_if}, {ClauseSet::CancelDirectiveName}}},
        {Version(51), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancel", Association::Unassociated, Category::Executable, {Clause::OMPC_if}, {ClauseSet::CancelDirectiveName}}},
        {Version(52), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancel", Association::Unassociated, Category::Executable, {Clause::OMPC_if}, {ClauseSet::CancelDirectiveName}}},
        {Version(60), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancel", Association::Unassociated, Category::Executable, {Clause::OMPC_if}, {ClauseSet::CancelDirectiveName}}},
        {Version(61), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancel", Association::Unassociated, Category::Executable, {Clause::OMPC_if}, {ClauseSet::CancelDirectiveName}}},
      }
    ),
  },
  {
    Directive::OMPD_cancellation_point,
    descriptor::Directive(
      "cancellation_point",
      {
        {Version(45), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancellation point", Association::Unassociated, Category::Executable, {Clause::OMPC_cancellation_construct_type}, {}}},
        {Version(50), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancellation point", Association::Unassociated, Category::Executable, {}, {ClauseSet::CancelDirectiveName}}},
        {Version(51), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancellation point", Association::Unassociated, Category::Executable, {}, {ClauseSet::CancelDirectiveName}}},
        {Version(52), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancellation point", Association::Unassociated, Category::Executable, {}, {ClauseSet::CancelDirectiveName}}},
        {Version(60), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancellation_point", Association::Unassociated, Category::Executable, {}, {ClauseSet::CancelDirectiveName}}},
        {Version(61), {{{Property::CancellationPoint, Property::ImplicitFlush}}, "cancellation_point", Association::Unassociated, Category::Executable, {}, {ClauseSet::CancelDirectiveName}}},
      }
    ),
  },
  {
    Directive::OMPD_critical,
    descriptor::Directive(
      "critical",
      {
        {Version(45), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::ThreadExclusive, Property::ThreadLimiting}}, "critical", Association::Block, Category::Executable, {Clause::OMPC_hint}, {}}},
        {Version(50), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::ThreadExclusive, Property::ThreadLimiting}}, "critical", Association::Block, Category::Executable, {Clause::OMPC_hint}, {}}},
        {Version(51), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::ThreadExclusive, Property::ThreadLimiting}}, "critical", Association::Block, Category::Executable, {Clause::OMPC_hint}, {}}},
        {Version(52), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::ThreadExclusive, Property::ThreadLimiting}}, "critical", Association::Block, Category::Executable, {Clause::OMPC_hint}, {}}},
        {Version(60), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::ThreadExclusive, Property::ThreadLimiting}}, "critical", Association::Block, Category::Executable, {Clause::OMPC_hint}, {}}},
        {Version(61), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::ThreadExclusive, Property::ThreadLimiting}}, "critical", Association::Block, Category::Executable, {Clause::OMPC_hint}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_declare_induction,
    descriptor::Directive(
      "declare_induction",
      {
        {Version(60), {{{Property::Pure}}, "declare_induction", Association::Unassociated, Category::Declarative, {Clause::OMPC_collector, Clause::OMPC_inductor}, {}}},
        {Version(61), {{{Property::Pure}}, "declare_induction", Association::Unassociated, Category::Declarative, {Clause::OMPC_collector, Clause::OMPC_inductor}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_declare_mapper,
    descriptor::Directive(
      "declare_mapper",
      {
        {Version(50), {{{Property::Pure}}, "declare mapper", Association::Unassociated, Category::Declarative, {Clause::OMPC_map}, {}}},
        {Version(51), {{{Property::Pure}}, "declare mapper", Association::Unassociated, Category::Declarative, {Clause::OMPC_map}, {}}},
        {Version(52), {{{Property::Pure}}, "declare mapper", Association::Unassociated, Category::Declarative, {Clause::OMPC_map}, {}}},
        {Version(60), {{{Property::Pure}}, "declare_mapper", Association::Unassociated, Category::Declarative, {Clause::OMPC_map}, {}}},
        {Version(61), {{{Property::Pure}}, "declare_mapper", Association::Unassociated, Category::Declarative, {Clause::OMPC_map}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_declare_reduction,
    descriptor::Directive(
      "declare_reduction",
      {
        {Version(45), {{{Property::Pure}}, "declare reduction", Association::Unassociated, Category::Declarative, {}, {}}},
        {Version(50), {{{Property::Pure}}, "declare reduction", Association::Unassociated, Category::Declarative, {Clause::OMPC_initializer}, {}}},
        {Version(51), {{{Property::Pure}}, "declare reduction", Association::Unassociated, Category::Declarative, {Clause::OMPC_initializer}, {}}},
        {Version(52), {{{Property::Pure}}, "declare reduction", Association::Unassociated, Category::Declarative, {Clause::OMPC_initializer}, {}}},
        {Version(60), {{{Property::Pure}}, "declare_reduction", Association::Unassociated, Category::Declarative, {Clause::OMPC_combiner, Clause::OMPC_initializer}, {}}},
        {Version(61), {{{Property::Pure}}, "declare_reduction", Association::Unassociated, Category::Declarative, {Clause::OMPC_combiner, Clause::OMPC_initializer, Clause::OMPC_reduction_kind}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_declare_simd,
    descriptor::Directive(
      "declare_simd",
      {
        {Version(45), {{{Property::Pure, Property::VariantGenerating}}, "declare simd", Association::Declaration, Category::Declarative, {Clause::OMPC_aligned, Clause::OMPC_linear, Clause::OMPC_simdlen, Clause::OMPC_uniform}, {ClauseSet::Branch}}},
        {Version(50), {{{Property::Pure, Property::VariantGenerating}}, "declare simd", Association::Declaration, Category::Declarative, {Clause::OMPC_aligned, Clause::OMPC_linear, Clause::OMPC_simdlen, Clause::OMPC_uniform}, {ClauseSet::Branch}}},
        {Version(51), {{{Property::Pure, Property::VariantGenerating}}, "declare simd", Association::Declaration, Category::Declarative, {Clause::OMPC_aligned, Clause::OMPC_linear, Clause::OMPC_simdlen, Clause::OMPC_uniform}, {ClauseSet::Branch}}},
        {Version(52), {{{Property::Pure, Property::VariantGenerating}}, "declare simd", Association::Declaration, Category::Declarative, {Clause::OMPC_aligned, Clause::OMPC_linear, Clause::OMPC_simdlen, Clause::OMPC_uniform}, {ClauseSet::Branch}}},
        {Version(60), {{{Property::Pure, Property::VariantGenerating}}, "declare_simd", Association::Declaration, Category::Declarative, {Clause::OMPC_aligned, Clause::OMPC_linear, Clause::OMPC_simdlen, Clause::OMPC_uniform}, {ClauseSet::Branch}}},
        {Version(61), {{{Property::Pure, Property::VariantGenerating}}, "declare_simd", Association::Declaration, Category::Declarative, {Clause::OMPC_aligned, Clause::OMPC_linear, Clause::OMPC_simdlen, Clause::OMPC_uniform}, {ClauseSet::Branch}}},
      }
    ),
  },
  {
    Directive::OMPD_declare_target,
    descriptor::Directive(
      "declare_target",
      {
        {Version(45), {{{Property::DeclareTarget, Property::Device, Property::Pure, Property::VariantGenerating}}, "declare target", Association::Explicit, Category::Declarative, {Clause::OMPC_link, Clause::OMPC_to}, {}}},
        {Version(50), {{{Property::DeclareTarget, Property::Device, Property::Pure, Property::VariantGenerating}}, "declare target", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_link, Clause::OMPC_to}, {}}},
        {Version(51), {{{Property::DeclareTarget, Property::Device, Property::Pure, Property::VariantGenerating}}, "declare target", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_indirect, Clause::OMPC_link, Clause::OMPC_to}, {}}},
        {Version(52), {{{Property::DeclareTarget, Property::Device, Property::Pure, Property::VariantGenerating}}, "declare target", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_enter, Clause::OMPC_indirect, Clause::OMPC_link}, {}}},
        {Version(60), {{{Property::DeclareTarget, Property::Device, Property::Pure, Property::VariantGenerating}}, "declare_target", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_enter, Clause::OMPC_indirect, Clause::OMPC_link, Clause::OMPC_local}, {}}},
        {Version(61), {{{Property::DeclareTarget, Property::Device, Property::Pure, Property::VariantGenerating}}, "declare_target", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type, Clause::OMPC_enter, Clause::OMPC_indirect, Clause::OMPC_link, Clause::OMPC_local}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_declare_variant,
    descriptor::Directive(
      "declare_variant",
      {
        {Version(50), {{{Property::Pure}}, "declare variant", Association::Declaration, Category::Declarative, {Clause::OMPC_match}, {}}},
        {Version(51), {{{Property::Pure}}, "declare variant", Association::Declaration, Category::Declarative, {Clause::OMPC_adjust_args, Clause::OMPC_append_args, Clause::OMPC_match}, {}}},
        {Version(52), {{{Property::Pure}}, "declare variant", Association::Declaration, Category::Declarative, {Clause::OMPC_adjust_args, Clause::OMPC_append_args, Clause::OMPC_match}, {}}},
        {Version(60), {{{Property::Pure}}, "declare_variant", Association::Declaration, Category::Declarative, {Clause::OMPC_adjust_args, Clause::OMPC_append_args, Clause::OMPC_match}, {}}},
        {Version(61), {{{Property::Pure}}, "declare_variant", Association::Declaration, Category::Declarative, {Clause::OMPC_adjust_args, Clause::OMPC_append_args, Clause::OMPC_match}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_depobj,
    descriptor::Directive(
      "depobj",
      {
        {Version(50), {{{}}, "depobj", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_update_depend_objects}, {ClauseSet::DepobjClauseSet1}}},
        {Version(51), {{{}}, "depobj", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_update_depend_objects}, {ClauseSet::DepobjClauseSet1}}},
        {Version(52), {{{}}, "depobj", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_update_depend_objects}, {ClauseSet::DepobjClauseSet1}}},
        {Version(60), {{{}}, "depobj", Association::Unassociated, Category::Executable, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_update_depend_objects}, {ClauseSet::DepobjClauseSet1}}},
        {Version(61), {{{}}, "depobj", Association::Unassociated, Category::Executable, {Clause::OMPC_destroy, Clause::OMPC_init, Clause::OMPC_update_depend_objects}, {ClauseSet::DepobjClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_dispatch,
    descriptor::Directive(
      "dispatch",
      {
        {Version(51), {{{Property::ContextMatching, Property::ImplicitTaskwait}}, "dispatch", Association::Block, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_is_device_ptr, Clause::OMPC_nocontext, Clause::OMPC_novariants, Clause::OMPC_nowait}, {}}},
        {Version(52), {{{Property::ContextMatching, Property::ImplicitTaskwait}}, "dispatch", Association::Block, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_is_device_ptr, Clause::OMPC_nocontext, Clause::OMPC_novariants, Clause::OMPC_nowait}, {}}},
        {Version(60), {{{Property::ContextMatching, Property::ImplicitTaskwait}}, "dispatch", Association::Block, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_has_device_addr, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_nocontext, Clause::OMPC_novariants, Clause::OMPC_nowait}, {}}},
        {Version(61), {{{Property::ContextMatching, Property::ImplicitTaskwait, Property::TaskSynchronizing}}, "dispatch", Association::Block, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_has_device_addr, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_nocontext, Clause::OMPC_novariants, Clause::OMPC_nowait}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_distribute,
    descriptor::Directive(
      "distribute",
      {
        {Version(45), {{{Property::Partitioned, Property::SimdPartitionable, Property::TeamsNestable, Property::WorkDistribution}}, "distribute", Association::LoopNest, Category::Executable, {Clause::OMPC_collapse, Clause::OMPC_dist_schedule, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_private}, {}}},
        {Version(50), {{{Property::Partitioned, Property::SimdPartitionable, Property::TeamsNestable, Property::WorkDistribution}}, "distribute", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_dist_schedule, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private}, {}}},
        {Version(51), {{{Property::Partitioned, Property::SimdPartitionable, Property::TeamsNestable, Property::WorkDistribution}}, "distribute", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_dist_schedule, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private}, {}}},
        {Version(52), {{{Property::Partitioned, Property::SimdPartitionable, Property::TeamsNestable, Property::WorkDistribution}}, "distribute", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_dist_schedule, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private}, {}}},
        {Version(60), {{{Property::Partitioned, Property::SimdPartitionable, Property::TeamsNestable, Property::WorkDistribution}}, "distribute", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_dist_schedule, Clause::OMPC_firstprivate, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private}, {}}},
        {Version(61), {{{Property::Partitioned, Property::SimdPartitionable, Property::TeamsNestable, Property::WorkDistribution}}, "distribute", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_dist_schedule, Clause::OMPC_firstprivate, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_do,
    descriptor::Directive(
      "do",
      {
        {Version(45), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "do", Association::LoopNest, Category::Executable, {Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(50), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "do", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(51), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "do", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(52), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "do", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(60), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "do", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(61), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TaskSynchronizing, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "do", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_error,
    descriptor::Directive(
      "error",
      {
        {Version(51), {{{Property::Pure}}, "error", Association::Unassociated, Category::Utility, {Clause::OMPC_at, Clause::OMPC_message, Clause::OMPC_severity}, {}}},
        {Version(52), {{{Property::Pure}}, "error", Association::Unassociated, Category::Utility, {Clause::OMPC_at, Clause::OMPC_message, Clause::OMPC_severity}, {}}},
        {Version(60), {{{Property::Pure}}, "error", Association::Unassociated, Category::Utility, {Clause::OMPC_at, Clause::OMPC_message, Clause::OMPC_severity}, {}}},
        {Version(61), {{{Property::Pure}}, "error", Association::Unassociated, Category::Utility, {Clause::OMPC_at, Clause::OMPC_message, Clause::OMPC_severity}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_flatten,
    descriptor::Directive(
      "flatten",
      {
        {Version(61), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "flatten", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_depth}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_flush,
    descriptor::Directive(
      "flush",
      {
        {Version(45), {{{}}, "flush", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(50), {{{}}, "flush", Association::Unassociated, Category::Executable, {}, {ClauseSet::MemoryOrder}}},
        {Version(51), {{{}}, "flush", Association::Unassociated, Category::Executable, {}, {ClauseSet::MemoryOrder}}},
        {Version(52), {{{}}, "flush", Association::Unassociated, Category::Executable, {}, {ClauseSet::MemoryOrder}}},
        {Version(60), {{{}}, "flush", Association::Unassociated, Category::Executable, {Clause::OMPC_memscope}, {ClauseSet::MemoryOrder}}},
        {Version(61), {{{}}, "flush", Association::Unassociated, Category::Executable, {Clause::OMPC_memscope}, {ClauseSet::MemoryOrder}}},
      }
    ),
  },
  {
    Directive::OMPD_for,
    descriptor::Directive(
      "for",
      {
        {Version(45), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "for", Association::LoopNest, Category::Executable, {Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(50), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "for", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(51), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "for", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(52), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "for", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(60), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "for", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
        {Version(61), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::SimdPartitionable, Property::TaskSynchronizing, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing, Property::WorksharingLoop}}, "for", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_firstprivate, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nowait, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_schedule}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_fuse,
    descriptor::Directive(
      "fuse",
      {
        {Version(60), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "fuse", Association::LoopSequence, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_looprange}, {}}},
        {Version(61), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "fuse", Association::LoopSequence, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_depth, Clause::OMPC_looprange}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_groupprivate,
    descriptor::Directive(
      "groupprivate",
      {
        {Version(60), {{{Property::Pure}}, "groupprivate", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type}, {}}},
        {Version(61), {{{Property::Pure}}, "groupprivate", Association::Explicit, Category::Declarative, {Clause::OMPC_device_type}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_interchange,
    descriptor::Directive(
      "interchange",
      {
        {Version(60), {{{Property::LoopTransforming, Property::NonrectangularCompatible, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "interchange", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_permutation}, {}}},
        {Version(61), {{{Property::LoopTransforming, Property::NonrectangularCompatible, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "interchange", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_permutation}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_interop,
    descriptor::Directive(
      "interop",
      {
        {Version(51), {{{Property::Device}}, "interop", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_device, Clause::OMPC_init, Clause::OMPC_nowait, Clause::OMPC_use}, {ClauseSet::ActionClause}}},
        {Version(52), {{{Property::Device}}, "interop", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_device, Clause::OMPC_init, Clause::OMPC_nowait, Clause::OMPC_use}, {ClauseSet::ActionClause}}},
        {Version(60), {{{Property::Device}}, "interop", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_device, Clause::OMPC_init, Clause::OMPC_nowait, Clause::OMPC_use}, {ClauseSet::ActionClause}}},
        {Version(61), {{{Property::Device, Property::TaskSynchronizing}}, "interop", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_device, Clause::OMPC_init, Clause::OMPC_nowait, Clause::OMPC_use}, {ClauseSet::ActionClause}}},
      }
    ),
  },
  {
    Directive::OMPD_loop,
    descriptor::Directive(
      "loop",
      {
        {Version(50), {{{Property::OrderConcurrentNestable, Property::Partitioned, Property::SequentiallyOrderable, Property::Simdizable, Property::TeamExecuted, Property::TeamsNestable, Property::WorkDistribution, Property::Worksharing}}, "loop", Association::LoopNest, Category::Executable, {Clause::OMPC_bind, Clause::OMPC_collapse, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(51), {{{Property::OrderConcurrentNestable, Property::Partitioned, Property::SequentiallyOrderable, Property::Simdizable, Property::TeamExecuted, Property::TeamsNestable, Property::WorkDistribution, Property::Worksharing}}, "loop", Association::LoopNest, Category::Executable, {Clause::OMPC_bind, Clause::OMPC_collapse, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(52), {{{Property::OrderConcurrentNestable, Property::Partitioned, Property::SequentiallyOrderable, Property::Simdizable, Property::TeamExecuted, Property::TeamsNestable, Property::WorkDistribution, Property::Worksharing}}, "loop", Association::LoopNest, Category::Executable, {Clause::OMPC_bind, Clause::OMPC_collapse, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(60), {{{Property::OrderConcurrentNestable, Property::Partitioned, Property::SequentiallyOrderable, Property::Simdizable, Property::TeamExecuted, Property::TeamsNestable, Property::WorkDistribution, Property::Worksharing}}, "loop", Association::LoopNest, Category::Executable, {Clause::OMPC_bind, Clause::OMPC_collapse, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(61), {{{Property::OrderConcurrentNestable, Property::Partitioned, Property::SequentiallyOrderable, Property::Simdizable, Property::TeamExecuted, Property::TeamsNestable, Property::WorkDistribution, Property::Worksharing}}, "loop", Association::LoopNest, Category::Executable, {Clause::OMPC_bind, Clause::OMPC_collapse, Clause::OMPC_lastprivate, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_masked,
    descriptor::Directive(
      "masked",
      {
        {Version(51), {{{Property::ThreadLimiting, Property::ThreadSelecting}}, "masked", Association::Block, Category::Executable, {Clause::OMPC_filter}, {}}},
        {Version(52), {{{Property::ThreadLimiting, Property::ThreadSelecting}}, "masked", Association::Block, Category::Executable, {Clause::OMPC_filter}, {}}},
        {Version(60), {{{Property::ThreadLimiting, Property::ThreadSelecting}}, "masked", Association::Block, Category::Executable, {Clause::OMPC_filter}, {}}},
        {Version(61), {{{Property::ThreadLimiting, Property::ThreadSelecting}}, "masked", Association::Block, Category::Executable, {Clause::OMPC_filter}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_master,
    descriptor::Directive(
      "master",
      {
        {Version(45), {{{Property::ThreadLimiting}}, "master", Association::Block, Category::Executable, {}, {}}},
        {Version(50), {{{Property::ThreadLimiting, Property::ThreadSelecting}}, "master", Association::Block, Category::Executable, {}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_metadirective,
    descriptor::Directive(
      "metadirective",
      {
        {Version(50), {{{Property::Pure}}, "metadirective", Association::Unassociated, Category::Meta, {Clause::OMPC_default_variant, Clause::OMPC_when}, {}}},
        {Version(51), {{{Property::Pure}}, "metadirective", Association::Unassociated, Category::Meta, {Clause::OMPC_default_variant, Clause::OMPC_when}, {}}},
        {Version(52), {{{Property::Pure}}, "metadirective", Association::Unassociated, Category::Meta, {Clause::OMPC_otherwise, Clause::OMPC_when}, {}}},
        {Version(60), {{{Property::Pure}}, "metadirective", Association::Unassociated, Category::Meta, {Clause::OMPC_otherwise, Clause::OMPC_when}, {}}},
        {Version(61), {{{Property::Pure}}, "metadirective", Association::Unassociated, Category::Meta, {Clause::OMPC_otherwise, Clause::OMPC_when}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_nothing,
    descriptor::Directive(
      "nothing",
      {
        {Version(51), {{{Property::LoopTransforming, Property::Pure}}, "nothing", Association::Unassociated, Category::Utility, {}, {}}},
        {Version(52), {{{Property::LoopTransforming, Property::Pure}}, "nothing", Association::Unassociated, Category::Utility, {}, {}}},
        {Version(60), {{{Property::LoopTransforming, Property::Pure}}, "nothing", Association::Unassociated, Category::Utility, {Clause::OMPC_apply}, {}}},
        {Version(61), {{{Property::LoopTransforming, Property::Pure}}, "nothing", Association::Unassociated, Category::Utility, {Clause::OMPC_apply}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_ordered_blockassoc,
    descriptor::Directive(
      "ordered-blockassoc",
      {
        {Version(45), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::Simdizable, Property::ThreadExclusive, Property::ThreadLimiting}}, "ordered", Association::Block, Category::Executable, {Clause::OMPC_simd, Clause::OMPC_threads}, {}}},
        {Version(50), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::Simdizable, Property::ThreadExclusive, Property::ThreadLimiting}}, "ordered", Association::Block, Category::Executable, {}, {ClauseSet::ParallelizationLevel}}},
        {Version(51), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::Simdizable, Property::ThreadExclusive, Property::ThreadLimiting}}, "ordered", Association::Block, Category::Executable, {}, {ClauseSet::ParallelizationLevel}}},
        {Version(52), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::Simdizable, Property::ThreadExclusive, Property::ThreadLimiting}}, "ordered", Association::Block, Category::Executable, {}, {ClauseSet::ParallelizationLevel}}},
        {Version(60), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::Simdizable, Property::ThreadExclusive, Property::ThreadLimiting}}, "ordered", Association::Block, Category::Executable, {}, {ClauseSet::ParallelizationLevel}}},
        {Version(61), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtExit, Property::MutualExclusion, Property::Simdizable, Property::ThreadExclusive, Property::ThreadLimiting}}, "ordered", Association::Block, Category::Executable, {}, {ClauseSet::ParallelizationLevel}}},
      }
    ),
  },
  {
    Directive::OMPD_ordered_standalone,
    descriptor::Directive(
      "ordered-standalone",
      {
        {Version(45), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::MutualExclusion}}, "ordered", Association::Unassociated, Category::Executable, {Clause::OMPC_depend}, {}}},
        {Version(50), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::MutualExclusion}}, "ordered", Association::Unassociated, Category::Executable, {Clause::OMPC_depend}, {}}},
        {Version(51), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::MutualExclusion}}, "ordered", Association::Unassociated, Category::Executable, {Clause::OMPC_depend}, {}}},
        {Version(52), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::MutualExclusion}}, "ordered", Association::Unassociated, Category::Executable, {Clause::OMPC_doacross}, {}}},
        {Version(60), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::MutualExclusion}}, "ordered", Association::Unassociated, Category::Executable, {Clause::OMPC_doacross}, {}}},
        {Version(61), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::MutualExclusion}}, "ordered", Association::Unassociated, Category::Executable, {Clause::OMPC_doacross}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_parallel,
    descriptor::Directive(
      "parallel",
      {
        {Version(45), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsNestable, Property::ThreadLimiting, Property::ThreadsOrderable}}, "parallel", Association::Block, Category::Executable, {Clause::OMPC_copyin, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_num_threads, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_reduction, Clause::OMPC_shared}, {}}},
        {Version(50), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsNestable, Property::ThreadLimiting, Property::ThreadsOrderable}}, "parallel", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyin, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_num_threads, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_reduction, Clause::OMPC_shared}, {}}},
        {Version(51), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsNestable, Property::ThreadLimiting, Property::ThreadsOrderable}}, "parallel", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyin, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_num_threads, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_reduction, Clause::OMPC_shared}, {}}},
        {Version(52), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsNestable, Property::ThreadLimiting, Property::ThreadsOrderable}}, "parallel", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyin, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_num_threads, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_reduction, Clause::OMPC_shared}, {}}},
        {Version(60), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsNestable, Property::ThreadLimiting, Property::ThreadsOrderable}}, "parallel", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyin, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_message, Clause::OMPC_num_threads, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_reduction, Clause::OMPC_safesync, Clause::OMPC_severity, Clause::OMPC_shared}, {}}},
        {Version(61), {{{Property::Cancellable, Property::CancellationPoint, Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSynchronizing, Property::TeamGenerating, Property::TeamsNestable, Property::ThreadLimiting, Property::ThreadsOrderable}}, "parallel", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyin, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_message, Clause::OMPC_num_threads, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_reduction, Clause::OMPC_safesync, Clause::OMPC_severity, Clause::OMPC_shared}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_requires,
    descriptor::Directive(
      "requires",
      {
        {Version(50), {{{}}, "requires", Association::Unassociated, Category::Informational, {}, {ClauseSet::Requirement}}},
        {Version(51), {{{}}, "requires", Association::Unassociated, Category::Informational, {}, {ClauseSet::Requirement}}},
        {Version(52), {{{}}, "requires", Association::Unassociated, Category::Informational, {}, {ClauseSet::Requirement}}},
        {Version(60), {{{}}, "requires", Association::Unassociated, Category::Informational, {}, {ClauseSet::Requirement}}},
        {Version(61), {{{}}, "requires", Association::Unassociated, Category::Informational, {}, {ClauseSet::Requirement}}},
      }
    ),
  },
  {
    Directive::OMPD_reverse,
    descriptor::Directive(
      "reverse",
      {
        {Version(60), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "reverse", Association::LoopNest, Category::Executable, {Clause::OMPC_apply}, {}}},
        {Version(61), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "reverse", Association::LoopNest, Category::Executable, {Clause::OMPC_apply}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_scan,
    descriptor::Directive(
      "scan",
      {
        {Version(50), {{{Property::Pure}}, "scan", Association::Separating, Category::Subsidiary, {Clause::OMPC_exclusive, Clause::OMPC_inclusive}, {ClauseSet::ScanClauseSet1}}},
        {Version(51), {{{Property::Pure}}, "scan", Association::Separating, Category::Subsidiary, {Clause::OMPC_exclusive, Clause::OMPC_inclusive}, {ClauseSet::ScanClauseSet1}}},
        {Version(52), {{{Property::Pure}}, "scan", Association::Separating, Category::Subsidiary, {Clause::OMPC_exclusive, Clause::OMPC_inclusive}, {ClauseSet::ScanClauseSet1}}},
        {Version(60), {{{Property::Pure}}, "scan", Association::Separating, Category::Subsidiary, {Clause::OMPC_exclusive, Clause::OMPC_inclusive, Clause::OMPC_init_complete}, {ClauseSet::ScanClauseSet1}}},
        {Version(61), {{{Property::Pure}}, "scan", Association::Separating, Category::Subsidiary, {Clause::OMPC_exclusive, Clause::OMPC_inclusive, Clause::OMPC_init_complete}, {ClauseSet::ScanClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_scope,
    descriptor::Directive(
      "scope",
      {
        {Version(51), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadsOrderable, Property::WorkDistribution, Property::Worksharing}}, "scope", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(52), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadsOrderable, Property::WorkDistribution, Property::Worksharing}}, "scope", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(60), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadsOrderable, Property::WorkDistribution, Property::Worksharing}}, "scope", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(61), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::TaskSynchronizing, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadsOrderable, Property::WorkDistribution, Property::Worksharing}}, "scope", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_section,
    descriptor::Directive(
      "section",
      {
        {Version(45), {{{}}, "section", Association::Separating, Category::Subsidiary, {}, {}}},
        {Version(50), {{{}}, "section", Association::Separating, Category::Subsidiary, {}, {}}},
        {Version(51), {{{}}, "section", Association::Separating, Category::Subsidiary, {}, {}}},
        {Version(52), {{{}}, "section", Association::Separating, Category::Subsidiary, {}, {}}},
        {Version(60), {{{}}, "section", Association::Separating, Category::Subsidiary, {}, {}}},
        {Version(61), {{{}}, "section", Association::Separating, Category::Subsidiary, {}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_sections,
    descriptor::Directive(
      "sections",
      {
        {Version(45), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::TeamExecuted, Property::ThreadLimiting, Property::WorkDistribution, Property::Worksharing}}, "sections", Association::Block, Category::Executable, {Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(50), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::TeamExecuted, Property::ThreadLimiting, Property::WorkDistribution, Property::Worksharing}}, "sections", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(51), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::TeamExecuted, Property::ThreadLimiting, Property::WorkDistribution, Property::Worksharing}}, "sections", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(52), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::TeamExecuted, Property::ThreadLimiting, Property::WorkDistribution, Property::Worksharing}}, "sections", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(60), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::TeamExecuted, Property::ThreadLimiting, Property::WorkDistribution, Property::Worksharing}}, "sections", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
        {Version(61), {{{Property::Cancellable, Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::SequentiallyOrderable, Property::TaskSynchronizing, Property::TeamExecuted, Property::ThreadLimiting, Property::WorkDistribution, Property::Worksharing}}, "sections", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate, Clause::OMPC_nowait, Clause::OMPC_private, Clause::OMPC_reduction}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_simd,
    descriptor::Directive(
      "simd",
      {
        {Version(45), {{{Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::Pure, Property::SequentiallyOrderable, Property::Simdizable}}, "simd", Association::LoopNest, Category::Executable, {Clause::OMPC_aligned, Clause::OMPC_collapse, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_safelen, Clause::OMPC_simdlen}, {}}},
        {Version(50), {{{Property::ContextMatching, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::Pure, Property::SequentiallyOrderable, Property::Simdizable}}, "simd", Association::LoopNest, Category::Executable, {Clause::OMPC_aligned, Clause::OMPC_collapse, Clause::OMPC_if, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nontemporal, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_safelen, Clause::OMPC_simdlen}, {}}},
        {Version(51), {{{Property::ContextMatching, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::Pure, Property::SequentiallyOrderable, Property::Simdizable}}, "simd", Association::LoopNest, Category::Executable, {Clause::OMPC_aligned, Clause::OMPC_collapse, Clause::OMPC_if, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nontemporal, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_safelen, Clause::OMPC_simdlen}, {}}},
        {Version(52), {{{Property::ContextMatching, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::Pure, Property::SequentiallyOrderable, Property::Simdizable}}, "simd", Association::LoopNest, Category::Executable, {Clause::OMPC_aligned, Clause::OMPC_collapse, Clause::OMPC_if, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nontemporal, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_safelen, Clause::OMPC_simdlen}, {}}},
        {Version(60), {{{Property::ContextMatching, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::Pure, Property::SequentiallyOrderable, Property::Simdizable}}, "simd", Association::LoopNest, Category::Executable, {Clause::OMPC_aligned, Clause::OMPC_collapse, Clause::OMPC_if, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nontemporal, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_safelen, Clause::OMPC_simdlen}, {}}},
        {Version(61), {{{Property::ContextMatching, Property::OrderConcurrentNestable, Property::ParallelismGenerating, Property::Pure, Property::SequentiallyOrderable, Property::Simdizable, Property::TeamsNestable}}, "simd", Association::LoopNest, Category::Executable, {Clause::OMPC_aligned, Clause::OMPC_collapse, Clause::OMPC_if, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_nontemporal, Clause::OMPC_order, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_safelen, Clause::OMPC_simdlen}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_single,
    descriptor::Directive(
      "single",
      {
        {Version(45), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadSelecting, Property::WorkDistribution, Property::Worksharing}}, "single", Association::Block, Category::Executable, {Clause::OMPC_copyprivate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private}, {ClauseSet::SingleClauseSet1}}},
        {Version(50), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadSelecting, Property::WorkDistribution, Property::Worksharing}}, "single", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyprivate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private}, {ClauseSet::SingleClauseSet1}}},
        {Version(51), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadSelecting, Property::WorkDistribution, Property::Worksharing}}, "single", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyprivate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private}, {ClauseSet::SingleClauseSet1}}},
        {Version(52), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadSelecting, Property::WorkDistribution, Property::Worksharing}}, "single", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyprivate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private}, {ClauseSet::SingleClauseSet1}}},
        {Version(60), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadSelecting, Property::WorkDistribution, Property::Worksharing}}, "single", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyprivate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private}, {ClauseSet::SingleClauseSet1}}},
        {Version(61), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TaskSynchronizing, Property::TeamExecuted, Property::ThreadLimiting, Property::ThreadSelecting, Property::WorkDistribution, Property::Worksharing}}, "single", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_copyprivate, Clause::OMPC_firstprivate, Clause::OMPC_nowait, Clause::OMPC_private}, {ClauseSet::SingleClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_split,
    descriptor::Directive(
      "split",
      {
        {Version(60), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "split", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_counts}, {}}},
        {Version(61), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "split", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_counts}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_stripe,
    descriptor::Directive(
      "stripe",
      {
        {Version(60), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "stripe", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_sizes}, {}}},
        {Version(61), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "stripe", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_sizes}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_target,
    descriptor::Directive(
      "target",
      {
        {Version(45), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExceptionAborting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TeamGenerating, Property::ThreadLimiting}}, "target", Association::Block, Category::Executable, {Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_is_device_ptr, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_ompx_bare, Clause::OMPC_private}, {}}},
        {Version(50), {{{Property::ContextMatching, Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExceptionAborting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TeamGenerating, Property::ThreadLimiting}}, "target", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_is_device_ptr, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_thread_limit, Clause::OMPC_uses_allocators}, {}}},
        {Version(51), {{{Property::ContextMatching, Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExceptionAborting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TeamGenerating, Property::ThreadLimiting}}, "target", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_firstprivate, Clause::OMPC_has_device_addr, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_is_device_ptr, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_thread_limit, Clause::OMPC_uses_allocators}, {}}},
        {Version(52), {{{Property::ContextMatching, Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExceptionAborting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TeamGenerating, Property::ThreadLimiting}}, "target", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_firstprivate, Clause::OMPC_has_device_addr, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_is_device_ptr, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_thread_limit, Clause::OMPC_uses_allocators}, {}}},
        {Version(60), {{{Property::ContextMatching, Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExceptionAborting, Property::ExplicitTaskGenerating, Property::GroupprivateInstantiating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TeamGenerating, Property::ThreadLimiting}}, "target", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_device_type, Clause::OMPC_firstprivate, Clause::OMPC_has_device_addr, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_is_device_ptr, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_ompx_bare, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_replayable, Clause::OMPC_thread_limit, Clause::OMPC_uses_allocators}, {}}},
        {Version(61), {{{Property::ContextMatching, Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExceptionAborting, Property::ExplicitTaskGenerating, Property::GroupprivateInstantiating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing, Property::TeamGenerating, Property::ThreadLimiting}}, "target", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_device_type, Clause::OMPC_dyn_groupprivate, Clause::OMPC_firstprivate, Clause::OMPC_has_device_addr, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_is_device_ptr, Clause::OMPC_map, Clause::OMPC_message, Clause::OMPC_nowait, Clause::OMPC_ompx_bare, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_replayable, Clause::OMPC_severity, Clause::OMPC_thread_limit, Clause::OMPC_uses_allocators}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_target_data,
    descriptor::Directive(
      "target_data",
      {
        {Version(45), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target data", Association::Block, Category::Executable, {Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_use_device_ptr}, {ClauseSet::TargetDataClauseSet1}}},
        {Version(50), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target data", Association::Block, Category::Executable, {Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {ClauseSet::DataEnvironmentClause}}},
        {Version(51), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target data", Association::Block, Category::Executable, {Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {ClauseSet::DataEnvironmentClause}}},
        {Version(52), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target data", Association::Block, Category::Executable, {Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {ClauseSet::DataEnvironmentClause}}},
        {Version(60), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::SharingTask, Property::SimpleComposite, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target_data", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_map, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_transparent, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {ClauseSet::DataEnvironmentClause}}},
        {Version(61), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::MapEntering, Property::MapExiting, Property::ParallelismGenerating, Property::SharingTask, Property::SimpleComposite, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing}}, "target_data", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_map, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_transparent, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr}, {ClauseSet::DataEnvironmentClause}}},
      }
    ),
  },
  {
    Directive::OMPD_target_enter_data,
    descriptor::Directive(
      "target_enter_data",
      {
        {Version(45), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target enter data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetEnterDataClauseSet1}}},
        {Version(50), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target enter data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetEnterDataClauseSet1}}},
        {Version(51), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target enter data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetEnterDataClauseSet1}}},
        {Version(52), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target enter data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetEnterDataClauseSet1}}},
        {Version(60), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target_enter_data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_replayable}, {ClauseSet::TargetEnterDataClauseSet1}}},
        {Version(61), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapEntering, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing}}, "target_enter_data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_replayable}, {ClauseSet::TargetEnterDataClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_target_exit_data,
    descriptor::Directive(
      "target_exit_data",
      {
        {Version(45), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target exit data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetExitDataClauseSet1}}},
        {Version(50), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target exit data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetExitDataClauseSet1}}},
        {Version(51), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target exit data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetExitDataClauseSet1}}},
        {Version(52), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target exit data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait}, {ClauseSet::TargetExitDataClauseSet1}}},
        {Version(60), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target_exit_data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_replayable}, {ClauseSet::TargetExitDataClauseSet1}}},
        {Version(61), {{{Property::DataMapping, Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::MapExiting, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing}}, "target_exit_data", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_if, Clause::OMPC_map, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_replayable}, {ClauseSet::TargetExitDataClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_target_update,
    descriptor::Directive(
      "target_update",
      {
        {Version(45), {{{Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target update", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_from, Clause::OMPC_if, Clause::OMPC_nowait, Clause::OMPC_to}, {ClauseSet::TargetUpdateClauseSet1}}},
        {Version(50), {{{Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target update", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_from, Clause::OMPC_if, Clause::OMPC_nowait, Clause::OMPC_to}, {ClauseSet::TargetUpdateClauseSet1}}},
        {Version(51), {{{Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target update", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_from, Clause::OMPC_if, Clause::OMPC_nowait, Clause::OMPC_to}, {ClauseSet::TargetUpdateClauseSet1}}},
        {Version(52), {{{Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target update", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_from, Clause::OMPC_if, Clause::OMPC_nowait, Clause::OMPC_to}, {ClauseSet::TargetUpdateClauseSet1}}},
        {Version(60), {{{Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "target_update", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_from, Clause::OMPC_if, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_replayable, Clause::OMPC_to}, {ClauseSet::TargetUpdateClauseSet1}}},
        {Version(61), {{{Property::Device, Property::DeviceAffecting, Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing}}, "target_update", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_device, Clause::OMPC_from, Clause::OMPC_if, Clause::OMPC_nowait, Clause::OMPC_priority, Clause::OMPC_replayable, Clause::OMPC_to}, {ClauseSet::TargetUpdateClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_task,
    descriptor::Directive(
      "task",
      {
        {Version(45), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::ThreadLimiting}}, "task", Association::Block, Category::Executable, {Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_mergeable, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_untied}, {}}},
        {Version(50), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::ThreadLimiting}}, "task", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_mergeable, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::TaskClauseSet1}}},
        {Version(51), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::ThreadLimiting}}, "task", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_mergeable, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::TaskClauseSet1}}},
        {Version(52), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::ThreadLimiting}}, "task", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_mergeable, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::TaskClauseSet1}}},
        {Version(60), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::ThreadLimiting}}, "task", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_mergeable, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_replayable, Clause::OMPC_shared, Clause::OMPC_threadset, Clause::OMPC_transparent, Clause::OMPC_untied}, {ClauseSet::TaskClauseSet1}}},
        {Version(61), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ParallelismGenerating, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing, Property::TeamsNestable, Property::ThreadLimiting}}, "task", Association::Block, Category::Executable, {Clause::OMPC_affinity, Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_depend, Clause::OMPC_detach, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_mergeable, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_replayable, Clause::OMPC_shared, Clause::OMPC_threadset, Clause::OMPC_transparent, Clause::OMPC_untied}, {ClauseSet::TaskClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_taskgraph,
    descriptor::Directive(
      "taskgraph",
      {
        {Version(60), {{{Property::ImplicitTaskgroup, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskgraph", Association::Block, Category::Executable, {Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_if, Clause::OMPC_nogroup, Clause::OMPC_priority}, {}}},
        {Version(61), {{{Property::ImplicitTaskgroup, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing}}, "taskgraph", Association::Block, Category::Executable, {Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_if, Clause::OMPC_nogroup, Clause::OMPC_priority}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_taskgroup,
    descriptor::Directive(
      "taskgroup",
      {
        {Version(45), {{{Property::Cancellable, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskgroup", Association::Block, Category::Executable, {}, {}}},
        {Version(50), {{{Property::Cancellable, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskgroup", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_task_reduction}, {}}},
        {Version(51), {{{Property::Cancellable, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskgroup", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_task_reduction}, {}}},
        {Version(52), {{{Property::Cancellable, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskgroup", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_task_reduction}, {}}},
        {Version(60), {{{Property::Cancellable, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskgroup", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_task_reduction}, {}}},
        {Version(61), {{{Property::Cancellable, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing, Property::TeamsNestable}}, "taskgroup", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_task_reduction}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_taskloop,
    descriptor::Directive(
      "taskloop",
      {
        {Version(45), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::ParallelismGenerating, Property::SimdPartitionable, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskloop", Association::LoopNest, Category::Executable, {Clause::OMPC_collapse, Clause::OMPC_default, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_grainsize, Clause::OMPC_if, Clause::OMPC_lastprivate, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_num_tasks, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::GranularityClause}}},
        {Version(50), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::ParallelismGenerating, Property::SimdPartitionable, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskloop", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_default, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_grainsize, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_lastprivate, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_num_tasks, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::GranularityClause, ClauseSet::SynchronizationClause}}},
        {Version(51), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::ParallelismGenerating, Property::SimdPartitionable, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskloop", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_default, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_grainsize, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_lastprivate, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_num_tasks, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::GranularityClause, ClauseSet::SynchronizationClause}}},
        {Version(52), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::ParallelismGenerating, Property::SimdPartitionable, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskloop", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_default, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_grainsize, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_lastprivate, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_num_tasks, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_untied}, {ClauseSet::GranularityClause, ClauseSet::SynchronizationClause}}},
        {Version(60), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::ParallelismGenerating, Property::SimdPartitionable, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskloop", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_default, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_grainsize, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_num_tasks, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_replayable, Clause::OMPC_shared, Clause::OMPC_threadset, Clause::OMPC_transparent, Clause::OMPC_untied}, {ClauseSet::GranularityClause, ClauseSet::SynchronizationClause}}},
        {Version(61), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskgroup, Property::ParallelismGenerating, Property::SimdPartitionable, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing, Property::TeamsNestable}}, "taskloop", Association::LoopNest, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_collapse, Clause::OMPC_default, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_grainsize, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_induction, Clause::OMPC_lastprivate, Clause::OMPC_mergeable, Clause::OMPC_nogroup, Clause::OMPC_num_tasks, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_replayable, Clause::OMPC_shared, Clause::OMPC_threadset, Clause::OMPC_transparent, Clause::OMPC_untied}, {ClauseSet::GranularityClause, ClauseSet::SynchronizationClause}}},
      }
    ),
  },
  {
    Directive::OMPD_taskwait,
    descriptor::Directive(
      "taskwait",
      {
        {Version(45), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskwait", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(50), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskwait", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_nowait}, {}}},
        {Version(51), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskwait", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_nowait}, {}}},
        {Version(52), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskwait", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_nowait}, {}}},
        {Version(60), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit}}, "taskwait", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_nowait, Clause::OMPC_replayable}, {}}},
        {Version(61), {{{Property::ExplicitTaskGenerating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::TaskGenerating, Property::TaskSchedulingPoint, Property::TaskSchedulingPointAfterTaskGeneration, Property::TaskSchedulingPointAtEntry, Property::TaskSchedulingPointAtExit, Property::TaskSynchronizing, Property::TeamsNestable}}, "taskwait", Association::Unassociated, Category::Executable, {Clause::OMPC_depend, Clause::OMPC_nowait, Clause::OMPC_replayable}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_taskyield,
    descriptor::Directive(
      "taskyield",
      {
        {Version(45), {{{Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskyield", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(50), {{{Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskyield", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(51), {{{Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskyield", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(52), {{{Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskyield", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(60), {{{Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit}}, "taskyield", Association::Unassociated, Category::Executable, {}, {}}},
        {Version(61), {{{Property::TaskSchedulingPoint, Property::TaskSchedulingPointAtExit, Property::TeamsNestable}}, "taskyield", Association::Unassociated, Category::Executable, {}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_teams,
    descriptor::Directive(
      "teams",
      {
        {Version(45), {{{Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::LeagueGenerating, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsOrderable, Property::ThreadLimiting}}, "teams", Association::Block, Category::Executable, {Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_num_teams, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_thread_limit}, {}}},
        {Version(50), {{{Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::LeagueGenerating, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsOrderable, Property::ThreadLimiting}}, "teams", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_num_teams, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_thread_limit}, {}}},
        {Version(51), {{{Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::LeagueGenerating, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsOrderable, Property::ThreadLimiting}}, "teams", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_num_teams, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_thread_limit}, {}}},
        {Version(52), {{{Property::ContextMatching, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::LeagueGenerating, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsOrderable, Property::ThreadLimiting}}, "teams", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_num_teams, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_thread_limit}, {}}},
        {Version(60), {{{Property::ContextMatching, Property::GroupprivateInstantiating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::LeagueGenerating, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsOrderable, Property::ThreadLimiting}}, "teams", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_num_teams, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_shared, Clause::OMPC_thread_limit}, {}}},
        {Version(61), {{{Property::ContextMatching, Property::GroupprivateInstantiating, Property::ImplicitAcquireFlush, Property::ImplicitAcquireFlushAtEntry, Property::ImplicitAcquireFlushAtExit, Property::ImplicitFlush, Property::ImplicitReleaseFlush, Property::ImplicitReleaseFlushAtEntry, Property::ImplicitReleaseFlushAtExit, Property::ImplicitTaskGenerating, Property::LeagueGenerating, Property::ParallelismGenerating, Property::TaskGenerating, Property::TeamGenerating, Property::TeamsOrderable, Property::ThreadLimiting}}, "teams", Association::Block, Category::Executable, {Clause::OMPC_allocate, Clause::OMPC_default, Clause::OMPC_dyn_groupprivate, Clause::OMPC_firstprivate, Clause::OMPC_if, Clause::OMPC_message, Clause::OMPC_num_teams, Clause::OMPC_ompx_bare, Clause::OMPC_private, Clause::OMPC_reduction, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_thread_limit}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_threadprivate,
    descriptor::Directive(
      "threadprivate",
      {
        {Version(45), {{{Property::Pure}}, "threadprivate", Association::Explicit, Category::Declarative, {}, {}}},
        {Version(50), {{{Property::Pure}}, "threadprivate", Association::Explicit, Category::Declarative, {}, {}}},
        {Version(51), {{{Property::Pure}}, "threadprivate", Association::Explicit, Category::Declarative, {}, {}}},
        {Version(52), {{{Property::Pure}}, "threadprivate", Association::Explicit, Category::Declarative, {}, {}}},
        {Version(60), {{{Property::Pure}}, "threadprivate", Association::Explicit, Category::Declarative, {}, {}}},
        {Version(61), {{{Property::Pure}}, "threadprivate", Association::Explicit, Category::Declarative, {}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_tile,
    descriptor::Directive(
      "tile",
      {
        {Version(51), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "tile", Association::LoopNest, Category::Executable, {Clause::OMPC_sizes}, {}}},
        {Version(52), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "tile", Association::LoopNest, Category::Executable, {Clause::OMPC_sizes}, {}}},
        {Version(60), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "tile", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_sizes}, {}}},
        {Version(61), {{{Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "tile", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_shift, Clause::OMPC_sizes}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_unroll,
    descriptor::Directive(
      "unroll",
      {
        {Version(51), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "unroll", Association::LoopNest, Category::Executable, {Clause::OMPC_full, Clause::OMPC_partial}, {ClauseSet::UnrollClauseSet1}}},
        {Version(52), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "unroll", Association::LoopNest, Category::Executable, {Clause::OMPC_full, Clause::OMPC_partial}, {ClauseSet::UnrollClauseSet1}}},
        {Version(60), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "unroll", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_full, Clause::OMPC_partial}, {ClauseSet::UnrollClauseSet1}}},
        {Version(61), {{{Property::GenerallyComposable, Property::LoopTransforming, Property::OrderConcurrentNestable, Property::Pure, Property::Simdizable, Property::TeamsNestable}}, "unroll", Association::LoopNest, Category::Executable, {Clause::OMPC_apply, Clause::OMPC_full, Clause::OMPC_partial}, {ClauseSet::UnrollClauseSet1}}},
      }
    ),
  },
  {
    Directive::OMPD_workdistribute,
    descriptor::Directive(
      "workdistribute",
      {
        {Version(60), {{{Property::Partitioned, Property::WorkDistribution}}, "workdistribute", Association::Block, Category::Executable, {}, {}}},
        {Version(61), {{{Property::Partitioned, Property::TeamsNestable, Property::WorkDistribution}}, "workdistribute", Association::Block, Category::Executable, {}, {}}},
      }
    ),
  },
  {
    Directive::OMPD_workshare,
    descriptor::Directive(
      "workshare",
      {
        {Version(45), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing}}, "workshare", Association::Block, Category::Executable, {Clause::OMPC_nowait}, {}}},
        {Version(50), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing}}, "workshare", Association::Block, Category::Executable, {Clause::OMPC_nowait}, {}}},
        {Version(51), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing}}, "workshare", Association::Block, Category::Executable, {Clause::OMPC_nowait}, {}}},
        {Version(52), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing}}, "workshare", Association::Block, Category::Executable, {Clause::OMPC_nowait}, {}}},
        {Version(60), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing}}, "workshare", Association::Block, Category::Executable, {Clause::OMPC_nowait}, {}}},
        {Version(61), {{{Property::CancellationPoint, Property::ImplicitBarrier, Property::ImplicitBarrierFlushes, Property::ImplicitFlush, Property::Partitioned, Property::TaskSynchronizing, Property::TeamExecuted, Property::WorkDistribution, Property::Worksharing}}, "workshare", Association::Block, Category::Executable, {Clause::OMPC_nowait}, {}}},
      }
    ),
  },
#endif // GEN_OMP_DIRECTIVE_DESCRIPTORS
#ifdef GEN_OMP_MODIFIER_DESCRIPTORS
  {
    Modifier::AccessGroup,
    descriptor::Modifier(
      "access-group",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_dyn_groupprivate}}},
      }
    ),
  },
  {
    Modifier::AdjustOp,
    descriptor::Modifier(
      "adjust-op",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::AlignModifier,
    descriptor::Modifier(
      "align-modifier",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::Alignment,
    descriptor::Modifier(
      "alignment",
      {
        {Version(45), {{{Property::Compatible, Property::Constant, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(50), {{{Property::Compatible, Property::Constant, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(51), {{{Property::Compatible, Property::Optional, Property::Positive, Property::RegionInvariant, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(52), {{{Property::Compatible, Property::Optional, Property::Positive, Property::RegionInvariant, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(60), {{{Property::Compatible, Property::Constant, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(61), {{{Property::Compatible, Property::Constant, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
      }
    ),
  },
  {
    Modifier::AllocatorComplexModifier,
    descriptor::Modifier(
      "allocator-complex-modifier",
      {
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::AllocatorSimpleModifier,
    descriptor::Modifier(
      "allocator-simple-modifier",
      {
        {Version(50), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(51), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(52), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(60), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::AlwaysModifier,
    descriptor::Modifier(
      "always-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::AttachModifier,
    descriptor::Modifier(
      "attach-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::AutomapModifier,
    descriptor::Modifier(
      "automap-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_enter}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_enter}}},
      }
    ),
  },
  {
    Modifier::ChunkModifier,
    descriptor::Modifier(
      "chunk-modifier",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
      }
    ),
  },
  {
    Modifier::CloseModifier,
    descriptor::Modifier(
      "close-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::ContextSelector,
    descriptor::Modifier(
      "context-selector",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
      }
    ),
  },
  {
    Modifier::DefaultModifier,
    descriptor::Modifier(
      "default-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::DeleteModifier,
    descriptor::Modifier(
      "delete-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::DependenceType,
    descriptor::Modifier(
      "dependence-type",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
      }
    ),
  },
  {
    Modifier::DepinfoModifier,
    descriptor::Modifier(
      "depinfo-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Complex, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_init}}},
        {Version(61), {{{Property::Compatible, Property::Complex, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::DepobjModifier,
    descriptor::Modifier(
      "depobj-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    Modifier::DeviceModifier,
    descriptor::Modifier(
      "device-modifier",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_device}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_device}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_device}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_device}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_device}}},
      }
    ),
  },
  {
    Modifier::DimsModifier,
    descriptor::Modifier(
      "dims-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_thread_limit}}},
      }
    ),
  },
  {
    Modifier::DirectiveNameModifier,
    descriptor::Modifier(
      "directive-name-modifier",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_if}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_if}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_if}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_if}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_adjust_args, Clause::OMPC_affinity, Clause::OMPC_align, Clause::OMPC_aligned, Clause::OMPC_allocate, Clause::OMPC_allocator, Clause::OMPC_append_args, Clause::OMPC_apply, Clause::OMPC_at, Clause::OMPC_atomic_default_mem_order, Clause::OMPC_bind, Clause::OMPC_cancellation_construct_type, Clause::OMPC_capture, Clause::OMPC_collapse, Clause::OMPC_collector, Clause::OMPC_combiner, Clause::OMPC_compare, Clause::OMPC_contains, Clause::OMPC_copyin, Clause::OMPC_copyprivate, Clause::OMPC_counts, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_device_safesync, Clause::OMPC_device_type, Clause::OMPC_dist_schedule, Clause::OMPC_doacross, Clause::OMPC_dynamic_allocators, Clause::OMPC_enter, Clause::OMPC_exclusive, Clause::OMPC_fail, Clause::OMPC_filter, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_from, Clause::OMPC_full, Clause::OMPC_grainsize, Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_has_device_addr, Clause::OMPC_hint, Clause::OMPC_holds, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_inbranch, Clause::OMPC_inclusive, Clause::OMPC_indirect, Clause::OMPC_induction, Clause::OMPC_inductor, Clause::OMPC_init, Clause::OMPC_init_complete, Clause::OMPC_initializer, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_link, Clause::OMPC_local, Clause::OMPC_looprange, Clause::OMPC_map, Clause::OMPC_match, Clause::OMPC_memscope, Clause::OMPC_mergeable, Clause::OMPC_message, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism, Clause::OMPC_nocontext, Clause::OMPC_nogroup, Clause::OMPC_nontemporal, Clause::OMPC_notinbranch, Clause::OMPC_novariants, Clause::OMPC_nowait, Clause::OMPC_num_tasks, Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_ompx_bare, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_otherwise, Clause::OMPC_partial, Clause::OMPC_permutation, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_read, Clause::OMPC_reduction, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_replayable, Clause::OMPC_reverse_offload, Clause::OMPC_safelen, Clause::OMPC_safesync, Clause::OMPC_schedule, Clause::OMPC_self_maps, Clause::OMPC_seq_cst, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_simd, Clause::OMPC_simdlen, Clause::OMPC_sizes, Clause::OMPC_task_reduction, Clause::OMPC_thread_limit, Clause::OMPC_threads, Clause::OMPC_threadset, Clause::OMPC_to, Clause::OMPC_transparent, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory, Clause::OMPC_uniform, Clause::OMPC_untied, Clause::OMPC_update, Clause::OMPC_update_depend_objects, Clause::OMPC_use, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr, Clause::OMPC_uses_allocators, Clause::OMPC_weak, Clause::OMPC_when, Clause::OMPC_write}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_adjust_args, Clause::OMPC_affinity, Clause::OMPC_align, Clause::OMPC_aligned, Clause::OMPC_allocate, Clause::OMPC_allocator, Clause::OMPC_append_args, Clause::OMPC_apply, Clause::OMPC_at, Clause::OMPC_atomic_default_mem_order, Clause::OMPC_bind, Clause::OMPC_cancellation_construct_type, Clause::OMPC_capture, Clause::OMPC_collapse, Clause::OMPC_collector, Clause::OMPC_combiner, Clause::OMPC_compare, Clause::OMPC_contains, Clause::OMPC_copyin, Clause::OMPC_copyprivate, Clause::OMPC_counts, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_depth, Clause::OMPC_destroy, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_device_safesync, Clause::OMPC_device_type, Clause::OMPC_dist_schedule, Clause::OMPC_doacross, Clause::OMPC_dyn_groupprivate, Clause::OMPC_dynamic_allocators, Clause::OMPC_enter, Clause::OMPC_exclusive, Clause::OMPC_fail, Clause::OMPC_filter, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_from, Clause::OMPC_full, Clause::OMPC_grainsize, Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_has_device_addr, Clause::OMPC_hint, Clause::OMPC_holds, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_inbranch, Clause::OMPC_inclusive, Clause::OMPC_indirect, Clause::OMPC_induction, Clause::OMPC_inductor, Clause::OMPC_init, Clause::OMPC_init_complete, Clause::OMPC_initializer, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_link, Clause::OMPC_local, Clause::OMPC_looprange, Clause::OMPC_map, Clause::OMPC_match, Clause::OMPC_memscope, Clause::OMPC_mergeable, Clause::OMPC_message, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism, Clause::OMPC_nocontext, Clause::OMPC_nogroup, Clause::OMPC_nontemporal, Clause::OMPC_notinbranch, Clause::OMPC_novariants, Clause::OMPC_nowait, Clause::OMPC_num_tasks, Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_ompx_bare, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_otherwise, Clause::OMPC_partial, Clause::OMPC_permutation, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_read, Clause::OMPC_reduction, Clause::OMPC_reduction_kind, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_replayable, Clause::OMPC_reverse_offload, Clause::OMPC_safelen, Clause::OMPC_safesync, Clause::OMPC_schedule, Clause::OMPC_self_maps, Clause::OMPC_seq_cst, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_shift, Clause::OMPC_simd, Clause::OMPC_simdlen, Clause::OMPC_sizes, Clause::OMPC_task_reduction, Clause::OMPC_thread_limit, Clause::OMPC_threads, Clause::OMPC_threadset, Clause::OMPC_to, Clause::OMPC_transparent, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory, Clause::OMPC_uniform, Clause::OMPC_untied, Clause::OMPC_update, Clause::OMPC_update_depend_objects, Clause::OMPC_use, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr, Clause::OMPC_uses_allocators, Clause::OMPC_weak, Clause::OMPC_when, Clause::OMPC_write}}},
      }
    ),
  },
  {
    Modifier::Expectation,
    descriptor::Modifier(
      "expectation",
      {
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::Fallback,
    descriptor::Modifier(
      "fallback",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_use_device_ptr}}},
      }
    ),
  },
  {
    Modifier::FallbackModifier,
    descriptor::Modifier(
      "fallback-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_dyn_groupprivate}}},
      }
    ),
  },
  {
    Modifier::InductionIdentifier,
    descriptor::Modifier(
      "induction-identifier",
      {
        {Version(60), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_induction}}},
        {Version(61), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::InductionModifier,
    descriptor::Modifier(
      "induction-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_induction}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::InscanModifier,
    descriptor::Modifier(
      "inscan-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::InteropType,
    descriptor::Modifier(
      "interop-type",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Repeatable, Property::Required}}, {Clause::OMPC_init}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Repeatable, Property::Required}}, {Clause::OMPC_init}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Repeatable}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::Iterator,
    descriptor::Modifier(
      "iterator",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::LastprivateModifier,
    descriptor::Modifier(
      "lastprivate-modifier",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_lastprivate}}},
      }
    ),
  },
  {
    Modifier::LinearModifier,
    descriptor::Modifier(
      "linear-modifier",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::LinearStep,
    descriptor::Modifier(
      "linear-step",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::LocModifier,
    descriptor::Modifier(
      "loc-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    Modifier::LoopModifier,
    descriptor::Modifier(
      "loop-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_apply}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_apply}}},
      }
    ),
  },
  {
    Modifier::LowerBound,
    descriptor::Modifier(
      "lower-bound",
      {
        {Version(50), {{{Property::Compatible, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(51), {{{Property::Compatible, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(52), {{{Property::Compatible, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(60), {{{Property::Compatible, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(61), {{{Property::Compatible, Property::Optional, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
      }
    ),
  },
  {
    Modifier::MapType,
    descriptor::Modifier(
      "map-type",
      {
        {Version(45), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(50), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(51), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(52), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::MapTypeModifier,
    descriptor::Modifier(
      "map-type-modifier",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Repeatable}}, {Clause::OMPC_map}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Repeatable}}, {Clause::OMPC_map}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Repeatable}}, {Clause::OMPC_map}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Repeatable}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::Mapper,
    descriptor::Modifier(
      "mapper",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::MemSpace,
    descriptor::Modifier(
      "mem-space",
      {
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_uses_allocators}}},
      }
    ),
  },
  {
    Modifier::MotionModifier,
    descriptor::Modifier(
      "motion-modifier",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::NeedDeviceAddr,
    descriptor::Modifier(
      "need-device-addr",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::NeedDevicePtr,
    descriptor::Modifier(
      "need-device-ptr",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::NoncommutativeModifier,
    descriptor::Modifier(
      "noncommutative-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::Nothing,
    descriptor::Modifier(
      "nothing",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::OmpxHoldModifier,
    descriptor::Modifier(
      "ompx-hold-modifier",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::OptionalModifier,
    descriptor::Modifier(
      "optional-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
  {
    Modifier::OrderModifier,
    descriptor::Modifier(
      "order-modifier",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_order}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_order}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_order}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_order}}},
      }
    ),
  },
  {
    Modifier::OrderingModifier,
    descriptor::Modifier(
      "ordering-modifier",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_schedule}}},
      }
    ),
  },
  {
    Modifier::OriginalSharingModifier,
    descriptor::Modifier(
      "original-sharing-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::PerThreadModifier,
    descriptor::Modifier(
      "per-thread-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::PreferType,
    descriptor::Modifier(
      "prefer-type",
      {
        {Version(60), {{{Property::Compatible, Property::Complex, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_init}}},
        {Version(61), {{{Property::Compatible, Property::Complex, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::Prescriptiveness,
    descriptor::Modifier(
      "prescriptiveness",
      {
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks, Clause::OMPC_num_threads}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
      }
    ),
  },
  {
    Modifier::PrescriptivenessModifier,
    descriptor::Modifier(
      "prescriptiveness-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_num_threads, Clause::OMPC_thread_limit}}},
      }
    ),
  },
  {
    Modifier::PresentModifier,
    descriptor::Modifier(
      "present-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::ReductionIdentifier,
    descriptor::Modifier(
      "reduction-identifier",
      {
        {Version(45), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(50), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(51), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(52), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(60), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(61), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
      }
    ),
  },
  {
    Modifier::ReductionModifier,
    descriptor::Modifier(
      "reduction-modifier",
      {
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::RefModifier,
    descriptor::Modifier(
      "ref-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::Saved,
    descriptor::Modifier(
      "saved",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_firstprivate}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_firstprivate}}},
      }
    ),
  },
  {
    Modifier::ScaledModifier,
    descriptor::Modifier(
      "scaled-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_simdlen}}},
      }
    ),
  },
  {
    Modifier::SelfModifier,
    descriptor::Modifier(
      "self-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::MapTypeModifying, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::SizesSelector,
    descriptor::Modifier(
      "sizes-selector",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_sizes}}},
      }
    ),
  },
  {
    Modifier::StepComplexModifier,
    descriptor::Modifier(
      "step-complex-modifier",
      {
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::StepModifier,
    descriptor::Modifier(
      "step-modifier",
      {
        {Version(60), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_induction}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::StepSimpleModifier,
    descriptor::Modifier(
      "step-simple-modifier",
      {
        {Version(52), {{{Property::Exclusive, Property::Free, Property::Optional, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(60), {{{Property::Exclusive, Property::Free, Property::Optional, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::TargetType,
    descriptor::Modifier(
      "target-type",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::TargetsyncType,
    descriptor::Modifier(
      "targetsync-type",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::TaskDependenceType,
    descriptor::Modifier(
      "task-dependence-type",
      {
        {Version(45), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(50), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(51), {{{Property::Compatible, Property::Optional, Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(52), {{{Property::Compatible, Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_update_depend_objects}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_update_depend_objects}}},
      }
    ),
  },
  {
    Modifier::TaskModifier,
    descriptor::Modifier(
      "task-modifier",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::TraitsArray,
    descriptor::Modifier(
      "traits-array",
      {
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_uses_allocators}}},
      }
    ),
  },
  {
    Modifier::VariableCategory,
    descriptor::Modifier(
      "variable-category",
      {
        {Version(45), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Clause::OMPC_defaultmap}}},
        {Version(50), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(51), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(52), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(60), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
#endif // GEN_OMP_MODIFIER_DESCRIPTORS
#ifdef GEN_OMP_MODIFIER_GROUP_DESCRIPTORS
  {
    ModifierSet::AdjustOp,
    descriptor::ModifierSet(
      "adjust-op",
      {
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Required, Property::Unique}}, {Modifier::NeedDeviceAddr, Modifier::NeedDevicePtr, Modifier::Nothing}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    ModifierSet::DependListType,
    descriptor::ModifierSet(
      "depend-list-type",
      {
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Modifier::DepobjModifier, Modifier::LocModifier}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    ModifierSet::InteropType,
    descriptor::ModifierSet(
      "interop-type",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Modifier::TargetType, Modifier::TargetsyncType}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    ModifierSet::MapTypeModifying,
    descriptor::ModifierSet(
      "map-type-modifying",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Optional, Property::Unique}}, {Modifier::AlwaysModifier, Modifier::CloseModifier, Modifier::DeleteModifier, Modifier::PresentModifier, Modifier::SelfModifier}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    ModifierSet::ReductionType,
    descriptor::ModifierSet(
      "reduction-type",
      {
        {Version(61), {{{Property::Exclusive, Property::Free, Property::Optional, Property::Unique}}, {Modifier::DefaultModifier, Modifier::InscanModifier, Modifier::NoncommutativeModifier, Modifier::TaskModifier}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    ModifierSet::VariableSelector,
    descriptor::ModifierSet(
      "variable-selector",
      {
        {Version(61), {{{Property::Compatible, Property::Free, Property::Required, Property::Unique}}, {Modifier::OptionalModifier, Modifier::VariableCategory}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
#endif // GEN_OMP_MODIFIER_GROUP_DESCRIPTORS
#ifdef GEN_OMP_MODIFIER_SET_DESCRIPTORS
  {
    ModifierSet::DependModifierSet1,
    descriptor::ModifierSet(
      "depend_modifier_set_1",
      {
        {Version(45), {{{Property::Exclusive, Property::Required}}, {Modifier::DependenceType, Modifier::TaskDependenceType}, {Clause::OMPC_depend}}},
        {Version(50), {{{Property::Exclusive, Property::Required}}, {Modifier::DependenceType, Modifier::TaskDependenceType}, {Clause::OMPC_depend}}},
        {Version(51), {{{Property::Exclusive, Property::Required}}, {Modifier::DependenceType, Modifier::TaskDependenceType}, {Clause::OMPC_depend}}},
        {Version(61), {{{Property::Exclusive}}, {Modifier::DepobjModifier, Modifier::TaskDependenceType}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    ModifierSet::NumTeamsModifierSet1,
    descriptor::ModifierSet(
      "num_teams_modifier_set_1",
      {
        {Version(61), {{{Property::Exclusive}}, {Modifier::DimsModifier, Modifier::LowerBound}, {Clause::OMPC_num_teams}}},
      }
    ),
  },
  {
    ModifierSet::ReductionModifierSet1,
    descriptor::ModifierSet(
      "reduction_modifier_set_1",
      {
        {Version(61), {{{Property::Exclusive}}, {Modifier::InscanModifier, Modifier::NoncommutativeModifier, Modifier::PerThreadModifier}, {Clause::OMPC_reduction}}},
      }
    ),
  },
#endif // GEN_OMP_MODIFIER_SET_DESCRIPTORS
    // clang-format on
