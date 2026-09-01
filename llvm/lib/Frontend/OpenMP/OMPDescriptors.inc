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
        {Version(51), {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_acq_rel,
    descriptor::Clause(
      "acq_rel",
      {
        {Version(50), {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_acquire,
    descriptor::Clause(
      "acquire",
      {
        {Version(50), {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_adjust_args,
    descriptor::Clause(
      "adjust_args",
      {
        {Version(51), {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp}}},
        {Version(52), {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp}}},
        {Version(60), {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_affinity,
    descriptor::Clause(
      "affinity",
      {
        {Version(50), {{{}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}}},
        {Version(51), {{{}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}}},
        {Version(52), {{{Property::Unique}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}}},
        {Version(60), {{{Property::TaskInherited}}, "affinity", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator}}},
        {Version(61), {{{Property::TaskInherited}}, "affinity", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator}}},
      }
    ),
  },
  {
    Clause::OMPC_align,
    descriptor::Clause(
      "align",
      {
        {Version(51), {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_aligned,
    descriptor::Clause(
      "aligned",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_allocate,
    descriptor::Clause(
      "allocate",
      {
        {Version(50), {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AllocatorSimpleModifier}}},
        {Version(51), {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorSimpleModifier}}},
        {Version(52), {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier}}},
        {Version(60), {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::AllPrivatizing, Property::TargetConsistent}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_allocator,
    descriptor::Clause(
      "allocator",
      {
        {Version(50), {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_append_args,
    descriptor::Clause(
      "append_args",
      {
        {Version(51), {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_apply,
    descriptor::Clause(
      "apply",
      {
        {Version(60), {{{}}, "apply", {Directive::OMPD_fuse, Directive::OMPD_interchange, Directive::OMPD_nothing, Directive::OMPD_reverse, Directive::OMPD_split, Directive::OMPD_stripe, Directive::OMPD_tile, Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LoopModifier}}},
        {Version(61), {{{}}, "apply", {Directive::OMPD_flatten, Directive::OMPD_fuse, Directive::OMPD_interchange, Directive::OMPD_nothing, Directive::OMPD_reverse, Directive::OMPD_split, Directive::OMPD_stripe, Directive::OMPD_tile, Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LoopModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_at,
    descriptor::Clause(
      "at",
      {
        {Version(51), {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_atomic_default_mem_order,
    descriptor::Clause(
      "atomic_default_mem_order",
      {
        {Version(50), {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_bind,
    descriptor::Clause(
      "bind",
      {
        {Version(50), {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_capture,
    descriptor::Clause(
      "capture",
      {
        {Version(45), {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_collapse,
    descriptor::Clause(
      "collapse",
      {
        {Version(45), {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_collector,
    descriptor::Clause(
      "collector",
      {
        {Version(60), {{{Property::Required, Property::Unique}}, "collector", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required, Property::Unique}}, "collector", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_combiner,
    descriptor::Clause(
      "combiner",
      {
        {Version(60), {{{Property::Required, Property::Unique}}, "combiner", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required, Property::Unique}}, "combiner", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_compare,
    descriptor::Clause(
      "compare",
      {
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_contains,
    descriptor::Clause(
      "contains",
      {
        {Version(51), {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_copyin,
    descriptor::Clause(
      "copyin",
      {
        {Version(45), {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_copyprivate,
    descriptor::Clause(
      "copyprivate",
      {
        {Version(45), {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_counts,
    descriptor::Clause(
      "counts",
      {
        {Version(60), {{{Property::Required, Property::Unique}}, "counts", {Directive::OMPD_split}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required, Property::Unique}}, "counts", {Directive::OMPD_split}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_default,
    descriptor::Clause(
      "default",
      {
        {Version(45), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(51), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(52), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(60), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::VariableCategory}}},
        {Version(61), {{{Property::DefaultAttribute, Property::PostModified}}, "default", {Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_default_variant,
    descriptor::Clause(
      "default-variant",
      {
        {Version(50), {{{Property::Ultimate, Property::Unique}}, "default-variant", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Ultimate, Property::Unique}}, "default-variant", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
      }
    ),
  },
  {
    Clause::OMPC_defaultmap,
    descriptor::Clause(
      "defaultmap",
      {
        {Version(45), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(50), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(51), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(52), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {Version(60), {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::VariableCategory}}},
        {Version(61), {{{Property::DefaultAttribute, Property::PostModified}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_depend,
    descriptor::Clause(
      "depend",
      {
        {Version(45), {{{}}, "depend", {Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::TaskDependenceType}}},
        {Version(50), {{{}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::Iterator, Modifier::TaskDependenceType}}},
        {Version(51), {{{Property::DispatchRequirement}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::Iterator, Modifier::TaskDependenceType}}},
        {Version(52), {{{Property::DispatchRequirement}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::TaskDependenceType}}},
        {Version(60), {{{Property::DispatchRequirement, Property::TaskInherited, Property::TaskgraphAltering}}, "depend", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::TaskDependenceType}}},
        {Version(61), {{{Property::DispatchRequirement, Property::TaskInherited, Property::TaskSynchronizing, Property::TaskgraphAltering}}, "depend", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::TaskDependenceType}}},
      }
    ),
  },
  {
    Clause::OMPC_depth,
    descriptor::Clause(
      "depth",
      {
        {Version(61), {{{Property::Unique}}, "depth", {Directive::OMPD_flatten, Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_destroy,
    descriptor::Clause(
      "destroy",
      {
        {Version(50), {{{}}, "destroy", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_detach,
    descriptor::Clause(
      "detach",
      {
        {Version(50), {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_device,
    descriptor::Clause(
      "device",
      {
        {Version(45), {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}}},
        {Version(51), {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}}},
        {Version(52), {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}}},
        {Version(60), {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_device_safesync,
    descriptor::Clause(
      "device_safesync",
      {
        {Version(60), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "device_safesync", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "device_safesync", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_device_type,
    descriptor::Clause(
      "device_type",
      {
        {Version(50), {{{Property::Unique}}, "device_type", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target, Directive::OMPD_groupprivate, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target, Directive::OMPD_groupprivate, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_dist_schedule,
    descriptor::Clause(
      "dist_schedule",
      {
        {Version(45), {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_doacross,
    descriptor::Clause(
      "doacross",
      {
        {Version(52), {{{Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType}}},
        {Version(60), {{{Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_dyn_groupprivate,
    descriptor::Clause(
      "dyn_groupprivate",
      {
        {Version(61), {{{Property::TargetConsistent}}, "dyn_groupprivate", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AccessGroup, Modifier::DirectiveNameModifier, Modifier::FallbackModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_dynamic_allocators,
    descriptor::Clause(
      "dynamic_allocators",
      {
        {Version(50), {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_enter,
    descriptor::Clause(
      "enter",
      {
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AutomapModifier, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AutomapModifier, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_exclusive,
    descriptor::Clause(
      "exclusive",
      {
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_fail,
    descriptor::Clause(
      "fail",
      {
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_filter,
    descriptor::Clause(
      "filter",
      {
        {Version(51), {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_final,
    descriptor::Clause(
      "final",
      {
        {Version(45), {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_firstprivate,
    descriptor::Clause(
      "firstprivate",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Saved}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Saved}}},
      }
    ),
  },
  {
    Clause::OMPC_from,
    descriptor::Clause(
      "from",
      {
        {Version(45), {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Mapper}}},
        {Version(51), {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::Mapper, Modifier::MotionModifier}}},
        {Version(52), {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Expectation, Modifier::Iterator, Modifier::Mapper}}},
        {Version(60), {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
        {Version(61), {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_full,
    descriptor::Clause(
      "full",
      {
        {Version(51), {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_grainsize,
    descriptor::Clause(
      "grainsize",
      {
        {Version(45), {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {Version(52), {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {Version(60), {{{Property::TaskgraphAltering, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
        {Version(61), {{{Property::TaskgraphAltering, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
      }
    ),
  },
  {
    Clause::OMPC_graph_id,
    descriptor::Clause(
      "graph_id",
      {
        {Version(60), {{{Property::Unique}}, "graph_id", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "graph_id", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_graph_reset,
    descriptor::Clause(
      "graph_reset",
      {
        {Version(60), {{{Property::Unique}}, "graph_reset", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "graph_reset", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_has_device_addr,
    descriptor::Clause(
      "has_device_addr",
      {
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_hint,
    descriptor::Clause(
      "hint",
      {
        {Version(45), {{{Property::Unique}}, "hint", {Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_holds,
    descriptor::Clause(
      "holds",
      {
        {Version(51), {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_if,
    descriptor::Clause(
      "if",
      {
        {Version(45), {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(50), {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(51), {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(52), {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(60), {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_in_reduction,
    descriptor::Clause(
      "in_reduction",
      {
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
      }
    ),
  },
  {
    Clause::OMPC_inbranch,
    descriptor::Clause(
      "inbranch",
      {
        {Version(45), {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_inclusive,
    descriptor::Clause(
      "inclusive",
      {
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_indirect,
    descriptor::Clause(
      "indirect",
      {
        {Version(51), {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_induction,
    descriptor::Clause(
      "induction",
      {
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "induction", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::InductionIdentifier, Modifier::InductionModifier, Modifier::StepModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "induction", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::InductionIdentifier, Modifier::InductionModifier, Modifier::StepModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_inductor,
    descriptor::Clause(
      "inductor",
      {
        {Version(60), {{{Property::Required, Property::Unique}}, "inductor", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required, Property::Unique}}, "inductor", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_init,
    descriptor::Clause(
      "init",
      {
        {Version(51), {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::InteropType}}},
        {Version(52), {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::InteropType}}},
        {Version(60), {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DepinfoModifier, Modifier::DirectiveNameModifier, Modifier::InteropType, Modifier::PreferType}}},
        {Version(61), {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DepinfoModifier, Modifier::DirectiveNameModifier, Modifier::PreferType}}},
      }
    ),
  },
  {
    Clause::OMPC_init_complete,
    descriptor::Clause(
      "init_complete",
      {
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "init_complete", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "init_complete", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_initializer,
    descriptor::Clause(
      "initializer",
      {
        {Version(50), {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_interop,
    descriptor::Clause(
      "interop",
      {
        {Version(60), {{{Property::DispatchRequirement, Property::Unique}}, "interop", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DispatchRequirement, Property::Unique}}, "interop", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_is_device_ptr,
    descriptor::Clause(
      "is_device_ptr",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_lastprivate,
    descriptor::Clause(
      "lastprivate",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LastprivateModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LastprivateModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_linear,
    descriptor::Clause(
      "linear",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_link,
    descriptor::Clause(
      "link",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_local,
    descriptor::Clause(
      "local",
      {
        {Version(60), {{{Property::DataEnvironmentAttribute}}, "local", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute}}, "local", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_looprange,
    descriptor::Clause(
      "looprange",
      {
        {Version(60), {{{Property::Unique}}, "looprange", {Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "looprange", {Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_map,
    descriptor::Clause(
      "map",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MapType, Modifier::MapTypeModifier, Modifier::OmpxHoldModifier}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlwaysModifier, Modifier::CloseModifier, Modifier::DeleteModifier, Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::MapType, Modifier::Mapper, Modifier::OmpxHoldModifier, Modifier::PresentModifier, Modifier::RefModifier, Modifier::SelfModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AttachModifier, Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::MapType, Modifier::Mapper, Modifier::OmpxHoldModifier, Modifier::RefModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_match,
    descriptor::Clause(
      "match",
      {
        {Version(50), {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_memscope,
    descriptor::Clause(
      "memscope",
      {
        {Version(60), {{{Property::Unique}}, "memscope", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "memscope", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_mergeable,
    descriptor::Clause(
      "mergeable",
      {
        {Version(45), {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "mergeable", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "mergeable", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_message,
    descriptor::Clause(
      "message",
      {
        {Version(51), {{{Property::Unique}}, "message", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "message", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "message", {Directive::OMPD_error, Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "message", {Directive::OMPD_error, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp,
    descriptor::Clause(
      "no_openmp",
      {
        {Version(51), {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp_constructs,
    descriptor::Clause(
      "no_openmp_constructs",
      {
        {Version(60), {{{Property::Unique}}, "no_openmp_constructs", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "no_openmp_constructs", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp_routines,
    descriptor::Clause(
      "no_openmp_routines",
      {
        {Version(51), {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_parallelism,
    descriptor::Clause(
      "no_parallelism",
      {
        {Version(51), {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nocontext,
    descriptor::Clause(
      "nocontext",
      {
        {Version(51), {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nogroup,
    descriptor::Clause(
      "nogroup",
      {
        {Version(45), {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_target_data, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::OutermostLeaf, Property::TaskSynchronizing, Property::Unique}}, "nogroup", {Directive::OMPD_target_data, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nontemporal,
    descriptor::Clause(
      "nontemporal",
      {
        {Version(50), {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_notinbranch,
    descriptor::Clause(
      "notinbranch",
      {
        {Version(45), {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_novariants,
    descriptor::Clause(
      "novariants",
      {
        {Version(51), {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nowait,
    descriptor::Clause(
      "nowait",
      {
        {Version(45), {{{Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::TaskSynchronizing, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_num_tasks,
    descriptor::Clause(
      "num_tasks",
      {
        {Version(45), {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {Version(52), {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {Version(60), {{{Property::TaskgraphAltering, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
        {Version(61), {{{Property::TaskgraphAltering, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
      }
    ),
  },
  {
    Clause::OMPC_num_teams,
    descriptor::Clause(
      "num_teams",
      {
        {Version(45), {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}}},
        {Version(51), {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}}},
        {Version(52), {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}}},
        {Version(60), {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LowerBound}}},
        {Version(61), {{{Property::SpaceConfiguring, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::LowerBound}}},
      }
    ),
  },
  {
    Clause::OMPC_num_threads,
    descriptor::Clause(
      "num_threads",
      {
        {Version(45), {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
        {Version(61), {{{Property::SpaceConfiguring, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::PrescriptivenessModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_order,
    descriptor::Clause(
      "order",
      {
        {Version(50), {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::OrderModifier}}},
        {Version(52), {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::OrderModifier}}},
        {Version(60), {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OrderModifier}}},
        {Version(61), {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OrderModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_ordered,
    descriptor::Clause(
      "ordered",
      {
        {Version(45), {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_otherwise,
    descriptor::Clause(
      "otherwise",
      {
        {Version(52), {{{Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_partial,
    descriptor::Clause(
      "partial",
      {
        {Version(51), {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_permutation,
    descriptor::Clause(
      "permutation",
      {
        {Version(60), {{{Property::Unique}}, "permutation", {Directive::OMPD_interchange}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "permutation", {Directive::OMPD_interchange}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_priority,
    descriptor::Clause(
      "priority",
      {
        {Version(45), {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "priority", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "priority", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_private,
    descriptor::Clause(
      "private",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_proc_bind,
    descriptor::Clause(
      "proc_bind",
      {
        {Version(45), {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_read,
    descriptor::Clause(
      "read",
      {
        {Version(45), {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_reduction,
    descriptor::Clause(
      "reduction",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OriginalSharingModifier, Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OriginalSharingModifier, Modifier::PerThreadModifier, Modifier::ReductionIdentifier}}},
      }
    ),
  },
  {
    Clause::OMPC_relaxed,
    descriptor::Clause(
      "relaxed",
      {
        {Version(50), {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_release,
    descriptor::Clause(
      "release",
      {
        {Version(50), {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_replayable,
    descriptor::Clause(
      "replayable",
      {
        {Version(60), {{{Property::Unique}}, "replayable", {Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "replayable", {Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_reverse_offload,
    descriptor::Clause(
      "reverse_offload",
      {
        {Version(50), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_safelen,
    descriptor::Clause(
      "safelen",
      {
        {Version(45), {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_safesync,
    descriptor::Clause(
      "safesync",
      {
        {Version(60), {{{Property::Unique}}, "safesync", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "safesync", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_schedule,
    descriptor::Clause(
      "schedule",
      {
        {Version(45), {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {Version(50), {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {Version(51), {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {Version(52), {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {Version(60), {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::DirectiveNameModifier, Modifier::OrderingModifier}}},
        {Version(61), {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::DirectiveNameModifier, Modifier::OrderingModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_self_maps,
    descriptor::Clause(
      "self_maps",
      {
        {Version(60), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "self_maps", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "self_maps", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_seq_cst,
    descriptor::Clause(
      "seq_cst",
      {
        {Version(45), {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_severity,
    descriptor::Clause(
      "severity",
      {
        {Version(51), {{{Property::Unique}}, "severity", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "severity", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "severity", {Directive::OMPD_error, Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "severity", {Directive::OMPD_error, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_shared,
    descriptor::Clause(
      "shared",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_simd,
    descriptor::Clause(
      "simd",
      {
        {Version(45), {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_simdlen,
    descriptor::Clause(
      "simdlen",
      {
        {Version(45), {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ScaledModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_sizes,
    descriptor::Clause(
      "sizes",
      {
        {Version(51), {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_stripe, Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_stripe, Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::SizesSelector}}},
      }
    ),
  },
  {
    Clause::OMPC_task_reduction,
    descriptor::Clause(
      "task_reduction",
      {
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
      }
    ),
  },
  {
    Clause::OMPC_thread_limit,
    descriptor::Clause(
      "thread_limit",
      {
        {Version(45), {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::IcvModifying, Property::SpaceConfiguring, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::PrescriptivenessModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_threads,
    descriptor::Clause(
      "threads",
      {
        {Version(45), {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_threadset,
    descriptor::Clause(
      "threadset",
      {
        {Version(60), {{{Property::Unique}}, "threadset", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "threadset", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_to,
    descriptor::Clause(
      "to",
      {
        {Version(45), {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Mapper}}},
        {Version(51), {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::Mapper, Modifier::MotionModifier}}},
        {Version(52), {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Expectation, Modifier::Iterator, Modifier::Mapper}}},
        {Version(60), {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
        {Version(61), {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_transparent,
    descriptor::Clause(
      "transparent",
      {
        {Version(60), {{{Property::Unique}}, "transparent", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "transparent", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_unified_address,
    descriptor::Clause(
      "unified_address",
      {
        {Version(50), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_unified_shared_memory,
    descriptor::Clause(
      "unified_shared_memory",
      {
        {Version(50), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_uniform,
    descriptor::Clause(
      "uniform",
      {
        {Version(45), {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_untied,
    descriptor::Clause(
      "untied",
      {
        {Version(45), {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_update,
    descriptor::Clause(
      "update",
      {
        {Version(45), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_update_depend_objects,
    descriptor::Clause(
      "update-depend_objects",
      {
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::TaskDependenceType}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::TaskDependenceType}}},
      }
    ),
  },
  {
    Clause::OMPC_use,
    descriptor::Clause(
      "use",
      {
        {Version(51), {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_use_device_addr,
    descriptor::Clause(
      "use_device_addr",
      {
        {Version(50), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_use_device_ptr,
    descriptor::Clause(
      "use_device_ptr",
      {
        {Version(45), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Fallback}}},
      }
    ),
  },
  {
    Clause::OMPC_uses_allocators,
    descriptor::Clause(
      "uses_allocators",
      {
        {Version(50), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MemSpace, Modifier::TraitsArray}}},
        {Version(60), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::MemSpace, Modifier::TraitsArray}}},
        {Version(61), {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::MemSpace, Modifier::TraitsArray}}},
      }
    ),
  },
  {
    Clause::OMPC_weak,
    descriptor::Clause(
      "weak",
      {
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_when,
    descriptor::Clause(
      "when",
      {
        {Version(50), {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}}},
        {Version(51), {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}}},
        {Version(52), {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}}},
        {Version(60), {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector, Modifier::DirectiveNameModifier}}},
        {Version(61), {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_write,
    descriptor::Clause(
      "write",
      {
        {Version(45), {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(50), {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(51), {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(52), {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {Version(60), {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {Version(61), {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
#endif // GEN_OMP_CLAUSE_DESCRIPTORS
#ifdef GEN_OMP_MODIFIER_DESCRIPTORS
  {
    Modifier::AccessGroup,
    descriptor::Modifier(
      "access-group",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_dyn_groupprivate}}},
      }
    ),
  },
  {
    Modifier::AdjustOp,
    descriptor::Modifier(
      "adjust-op",
      {
        {Version(51), {{{Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
        {Version(52), {{{Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
        {Version(60), {{{Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::AlignModifier,
    descriptor::Modifier(
      "align-modifier",
      {
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::Alignment,
    descriptor::Modifier(
      "alignment",
      {
        {Version(45), {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(50), {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(51), {{{Property::Positive, Property::RegionInvariant, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(52), {{{Property::Positive, Property::RegionInvariant, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(60), {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {Version(61), {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
      }
    ),
  },
  {
    Modifier::AllocatorComplexModifier,
    descriptor::Modifier(
      "allocator-complex-modifier",
      {
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::AllocatorSimpleModifier,
    descriptor::Modifier(
      "allocator-simple-modifier",
      {
        {Version(50), {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(51), {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(52), {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(60), {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {Version(61), {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::AlwaysModifier,
    descriptor::Modifier(
      "always-modifier",
      {
        {Version(60), {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::AttachModifier,
    descriptor::Modifier(
      "attach-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::AutomapModifier,
    descriptor::Modifier(
      "automap-modifier",
      {
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_enter}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_enter}}},
      }
    ),
  },
  {
    Modifier::ChunkModifier,
    descriptor::Modifier(
      "chunk-modifier",
      {
        {Version(45), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
      }
    ),
  },
  {
    Modifier::CloseModifier,
    descriptor::Modifier(
      "close-modifier",
      {
        {Version(60), {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::ContextSelector,
    descriptor::Modifier(
      "context-selector",
      {
        {Version(50), {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(51), {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(52), {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(60), {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {Version(61), {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
      }
    ),
  },
  {
    Modifier::DefaultModifier,
    descriptor::Modifier(
      "default-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::DeleteModifier,
    descriptor::Modifier(
      "delete-modifier",
      {
        {Version(60), {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::DependenceType,
    descriptor::Modifier(
      "dependence-type",
      {
        {Version(45), {{{Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(52), {{{Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
        {Version(60), {{{Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
        {Version(61), {{{Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
      }
    ),
  },
  {
    Modifier::DepinfoModifier,
    descriptor::Modifier(
      "depinfo-modifier",
      {
        {Version(60), {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
        {Version(61), {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::DepobjModifier,
    descriptor::Modifier(
      "depobj-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    Modifier::DeviceModifier,
    descriptor::Modifier(
      "device-modifier",
      {
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_device}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_device}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_device}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_device}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_device}}},
      }
    ),
  },
  {
    Modifier::DimsModifier,
    descriptor::Modifier(
      "dims-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_thread_limit}}},
      }
    ),
  },
  {
    Modifier::DirectiveNameModifier,
    descriptor::Modifier(
      "directive-name-modifier",
      {
        {Version(45), {{{Property::Unique}}, {Clause::OMPC_if}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_if}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_if}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_if}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_adjust_args, Clause::OMPC_affinity, Clause::OMPC_align, Clause::OMPC_aligned, Clause::OMPC_allocate, Clause::OMPC_allocator, Clause::OMPC_append_args, Clause::OMPC_apply, Clause::OMPC_at, Clause::OMPC_atomic_default_mem_order, Clause::OMPC_bind, Clause::OMPC_capture, Clause::OMPC_collapse, Clause::OMPC_collector, Clause::OMPC_combiner, Clause::OMPC_compare, Clause::OMPC_contains, Clause::OMPC_copyin, Clause::OMPC_copyprivate, Clause::OMPC_counts, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_device_safesync, Clause::OMPC_device_type, Clause::OMPC_dist_schedule, Clause::OMPC_doacross, Clause::OMPC_dynamic_allocators, Clause::OMPC_enter, Clause::OMPC_exclusive, Clause::OMPC_fail, Clause::OMPC_filter, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_from, Clause::OMPC_full, Clause::OMPC_grainsize, Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_has_device_addr, Clause::OMPC_hint, Clause::OMPC_holds, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_inbranch, Clause::OMPC_inclusive, Clause::OMPC_indirect, Clause::OMPC_induction, Clause::OMPC_inductor, Clause::OMPC_init, Clause::OMPC_init_complete, Clause::OMPC_initializer, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_link, Clause::OMPC_local, Clause::OMPC_looprange, Clause::OMPC_map, Clause::OMPC_match, Clause::OMPC_memscope, Clause::OMPC_mergeable, Clause::OMPC_message, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism, Clause::OMPC_nocontext, Clause::OMPC_nogroup, Clause::OMPC_nontemporal, Clause::OMPC_notinbranch, Clause::OMPC_novariants, Clause::OMPC_nowait, Clause::OMPC_num_tasks, Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_otherwise, Clause::OMPC_partial, Clause::OMPC_permutation, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_read, Clause::OMPC_reduction, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_replayable, Clause::OMPC_reverse_offload, Clause::OMPC_safelen, Clause::OMPC_safesync, Clause::OMPC_schedule, Clause::OMPC_self_maps, Clause::OMPC_seq_cst, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_simd, Clause::OMPC_simdlen, Clause::OMPC_sizes, Clause::OMPC_task_reduction, Clause::OMPC_thread_limit, Clause::OMPC_threads, Clause::OMPC_threadset, Clause::OMPC_to, Clause::OMPC_transparent, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory, Clause::OMPC_uniform, Clause::OMPC_untied, Clause::OMPC_update, Clause::OMPC_update_depend_objects, Clause::OMPC_use, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr, Clause::OMPC_uses_allocators, Clause::OMPC_weak, Clause::OMPC_when, Clause::OMPC_write}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_adjust_args, Clause::OMPC_affinity, Clause::OMPC_align, Clause::OMPC_aligned, Clause::OMPC_allocate, Clause::OMPC_allocator, Clause::OMPC_append_args, Clause::OMPC_apply, Clause::OMPC_at, Clause::OMPC_atomic_default_mem_order, Clause::OMPC_bind, Clause::OMPC_capture, Clause::OMPC_collapse, Clause::OMPC_collector, Clause::OMPC_combiner, Clause::OMPC_compare, Clause::OMPC_contains, Clause::OMPC_copyin, Clause::OMPC_copyprivate, Clause::OMPC_counts, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_depth, Clause::OMPC_destroy, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_device_safesync, Clause::OMPC_device_type, Clause::OMPC_dist_schedule, Clause::OMPC_doacross, Clause::OMPC_dyn_groupprivate, Clause::OMPC_dynamic_allocators, Clause::OMPC_enter, Clause::OMPC_exclusive, Clause::OMPC_fail, Clause::OMPC_filter, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_from, Clause::OMPC_full, Clause::OMPC_grainsize, Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_has_device_addr, Clause::OMPC_hint, Clause::OMPC_holds, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_inbranch, Clause::OMPC_inclusive, Clause::OMPC_indirect, Clause::OMPC_induction, Clause::OMPC_inductor, Clause::OMPC_init, Clause::OMPC_init_complete, Clause::OMPC_initializer, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_link, Clause::OMPC_local, Clause::OMPC_looprange, Clause::OMPC_map, Clause::OMPC_match, Clause::OMPC_memscope, Clause::OMPC_mergeable, Clause::OMPC_message, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism, Clause::OMPC_nocontext, Clause::OMPC_nogroup, Clause::OMPC_nontemporal, Clause::OMPC_notinbranch, Clause::OMPC_novariants, Clause::OMPC_nowait, Clause::OMPC_num_tasks, Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_otherwise, Clause::OMPC_partial, Clause::OMPC_permutation, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_read, Clause::OMPC_reduction, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_replayable, Clause::OMPC_reverse_offload, Clause::OMPC_safelen, Clause::OMPC_safesync, Clause::OMPC_schedule, Clause::OMPC_self_maps, Clause::OMPC_seq_cst, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_simd, Clause::OMPC_simdlen, Clause::OMPC_sizes, Clause::OMPC_task_reduction, Clause::OMPC_thread_limit, Clause::OMPC_threads, Clause::OMPC_threadset, Clause::OMPC_to, Clause::OMPC_transparent, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory, Clause::OMPC_uniform, Clause::OMPC_untied, Clause::OMPC_update, Clause::OMPC_update_depend_objects, Clause::OMPC_use, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr, Clause::OMPC_uses_allocators, Clause::OMPC_weak, Clause::OMPC_when, Clause::OMPC_write}}},
      }
    ),
  },
  {
    Modifier::Expectation,
    descriptor::Modifier(
      "expectation",
      {
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::Fallback,
    descriptor::Modifier(
      "fallback",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_use_device_ptr}}},
      }
    ),
  },
  {
    Modifier::FallbackModifier,
    descriptor::Modifier(
      "fallback-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_dyn_groupprivate}}},
      }
    ),
  },
  {
    Modifier::InductionIdentifier,
    descriptor::Modifier(
      "induction-identifier",
      {
        {Version(60), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_induction}}},
        {Version(61), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::InductionModifier,
    descriptor::Modifier(
      "induction-modifier",
      {
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_induction}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::InscanModifier,
    descriptor::Modifier(
      "inscan-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::InteropType,
    descriptor::Modifier(
      "interop-type",
      {
        {Version(51), {{{Property::Repeatable, Property::Required}}, {Clause::OMPC_init}}},
        {Version(52), {{{Property::Repeatable, Property::Required}}, {Clause::OMPC_init}}},
        {Version(60), {{{Property::Repeatable}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::Iterator,
    descriptor::Modifier(
      "iterator",
      {
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::LastprivateModifier,
    descriptor::Modifier(
      "lastprivate-modifier",
      {
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
      }
    ),
  },
  {
    Modifier::LinearModifier,
    descriptor::Modifier(
      "linear-modifier",
      {
        {Version(45), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::LinearStep,
    descriptor::Modifier(
      "linear-step",
      {
        {Version(45), {{{Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(50), {{{Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(51), {{{Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::LocModifier,
    descriptor::Modifier(
      "loc-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    Modifier::LoopModifier,
    descriptor::Modifier(
      "loop-modifier",
      {
        {Version(60), {{{Property::Optional, Property::Unique}}, {Clause::OMPC_apply}}},
        {Version(61), {{{Property::Optional, Property::Unique}}, {Clause::OMPC_apply}}},
      }
    ),
  },
  {
    Modifier::LowerBound,
    descriptor::Modifier(
      "lower-bound",
      {
        {Version(50), {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(51), {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(52), {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(60), {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {Version(61), {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
      }
    ),
  },
  {
    Modifier::MapType,
    descriptor::Modifier(
      "map-type",
      {
        {Version(45), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(50), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(51), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(52), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::MapTypeModifier,
    descriptor::Modifier(
      "map-type-modifier",
      {
        {Version(45), {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
        {Version(50), {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
        {Version(51), {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
        {Version(52), {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::Mapper,
    descriptor::Modifier(
      "mapper",
      {
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::MemSpace,
    descriptor::Modifier(
      "mem-space",
      {
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
      }
    ),
  },
  {
    Modifier::MotionModifier,
    descriptor::Modifier(
      "motion-modifier",
      {
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::NeedDeviceAddr,
    descriptor::Modifier(
      "need-device-addr",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::NeedDevicePtr,
    descriptor::Modifier(
      "need-device-ptr",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::NoncommutativeModifier,
    descriptor::Modifier(
      "noncommutative-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::Nothing,
    descriptor::Modifier(
      "nothing",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::OmpxHoldModifier,
    descriptor::Modifier(
      "ompx-hold-modifier",
      {
        {Version(45), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::OptionalModifier,
    descriptor::Modifier(
      "optional-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
  {
    Modifier::OrderModifier,
    descriptor::Modifier(
      "order-modifier",
      {
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_order}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_order}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_order}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_order}}},
      }
    ),
  },
  {
    Modifier::OrderingModifier,
    descriptor::Modifier(
      "ordering-modifier",
      {
        {Version(45), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_schedule}}},
      }
    ),
  },
  {
    Modifier::OriginalSharingModifier,
    descriptor::Modifier(
      "original-sharing-modifier",
      {
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::PerThreadModifier,
    descriptor::Modifier(
      "per-thread-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::PreferType,
    descriptor::Modifier(
      "prefer-type",
      {
        {Version(60), {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
        {Version(61), {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::Prescriptiveness,
    descriptor::Modifier(
      "prescriptiveness",
      {
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks, Clause::OMPC_num_threads}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
      }
    ),
  },
  {
    Modifier::PrescriptivenessModifier,
    descriptor::Modifier(
      "prescriptiveness-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_num_threads, Clause::OMPC_thread_limit}}},
      }
    ),
  },
  {
    Modifier::PresentModifier,
    descriptor::Modifier(
      "present-modifier",
      {
        {Version(60), {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::ReductionIdentifier,
    descriptor::Modifier(
      "reduction-identifier",
      {
        {Version(45), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(50), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(51), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(52), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(60), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {Version(61), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
      }
    ),
  },
  {
    Modifier::ReductionModifier,
    descriptor::Modifier(
      "reduction-modifier",
      {
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::RefModifier,
    descriptor::Modifier(
      "ref-modifier",
      {
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::Saved,
    descriptor::Modifier(
      "saved",
      {
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_firstprivate}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_firstprivate}}},
      }
    ),
  },
  {
    Modifier::ScaledModifier,
    descriptor::Modifier(
      "scaled-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_simdlen}}},
      }
    ),
  },
  {
    Modifier::SelfModifier,
    descriptor::Modifier(
      "self-modifier",
      {
        {Version(60), {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::SizesSelector,
    descriptor::Modifier(
      "sizes-selector",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_sizes}}},
      }
    ),
  },
  {
    Modifier::StepComplexModifier,
    descriptor::Modifier(
      "step-complex-modifier",
      {
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::StepModifier,
    descriptor::Modifier(
      "step-modifier",
      {
        {Version(52), {{{Property::Required, Property::Unique}}, {}}},
        {Version(60), {{{Property::Required, Property::Unique}}, {Clause::OMPC_induction}}},
        {Version(61), {{{Property::Required, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::StepSimpleModifier,
    descriptor::Modifier(
      "step-simple-modifier",
      {
        {Version(52), {{{Property::Exclusive, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(60), {{{Property::Exclusive, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {Version(61), {{{Property::Exclusive, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::TargetType,
    descriptor::Modifier(
      "target-type",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::TargetsyncType,
    descriptor::Modifier(
      "targetsync-type",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::TaskDependenceType,
    descriptor::Modifier(
      "task-dependence-type",
      {
        {Version(45), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(50), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(51), {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(52), {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_update_depend_objects}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_update_depend_objects}}},
      }
    ),
  },
  {
    Modifier::TaskModifier,
    descriptor::Modifier(
      "task-modifier",
      {
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::TraitsArray,
    descriptor::Modifier(
      "traits-array",
      {
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
      }
    ),
  },
  {
    Modifier::VariableCategory,
    descriptor::Modifier(
      "variable-category",
      {
        {Version(45), {{{Property::Required, Property::Unique}}, {Clause::OMPC_defaultmap}}},
        {Version(50), {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(51), {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(52), {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(60), {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {Version(61), {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
#endif // GEN_OMP_MODIFIER_DESCRIPTORS
    // clang-format on
