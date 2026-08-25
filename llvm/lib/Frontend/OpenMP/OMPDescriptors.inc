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
        {51, {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "absent", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_acq_rel,
    descriptor::Clause(
      "acq_rel",
      {
        {50, {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "acq_rel", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_acquire,
    descriptor::Clause(
      "acquire",
      {
        {50, {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "acquire", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_adjust_args,
    descriptor::Clause(
      "adjust_args",
      {
        {51, {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp}}},
        {52, {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp}}},
        {60, {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AdjustOp, Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DispatchCompliant}}, "adjust_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_affinity,
    descriptor::Clause(
      "affinity",
      {
        {50, {{{}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}}},
        {51, {{{}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}}},
        {52, {{{Property::Unique}}, "affinity", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator}}},
        {60, {{{Property::TaskInherited}}, "affinity", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator}}},
        {61, {{{Property::TaskInherited}}, "affinity", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator}}},
      }
    ),
  },
  {
    Clause::OMPC_align,
    descriptor::Clause(
      "align",
      {
        {51, {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "align", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_aligned,
    descriptor::Clause(
      "aligned",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment, Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::PostModified}}, "aligned", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Alignment, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_allocate,
    descriptor::Clause(
      "allocate",
      {
        {50, {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AllocatorSimpleModifier}}},
        {51, {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorSimpleModifier}}},
        {52, {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier}}},
        {60, {{{Property::AllPrivatizing}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier, Modifier::DirectiveNameModifier}}},
        {61, {{{Property::AllPrivatizing, Property::TargetConsistent}}, "allocate", {Directive::OMPD_allocators, Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskgroup, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlignModifier, Modifier::AllocatorComplexModifier, Modifier::AllocatorSimpleModifier, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_allocator,
    descriptor::Clause(
      "allocator",
      {
        {50, {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::IcvDefaulted, Property::Unique}}, "allocator", {Directive::OMPD_allocate}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_append_args,
    descriptor::Clause(
      "append_args",
      {
        {51, {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DispatchCompliant, Property::Unique}}, "append_args", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_apply,
    descriptor::Clause(
      "apply",
      {
        {60, {{{}}, "apply", {Directive::OMPD_fuse, Directive::OMPD_interchange, Directive::OMPD_nothing, Directive::OMPD_reverse, Directive::OMPD_split, Directive::OMPD_stripe, Directive::OMPD_tile, Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LoopModifier}}},
        {61, {{{}}, "apply", {Directive::OMPD_flatten, Directive::OMPD_fuse, Directive::OMPD_interchange, Directive::OMPD_nothing, Directive::OMPD_reverse, Directive::OMPD_split, Directive::OMPD_stripe, Directive::OMPD_tile, Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LoopModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_at,
    descriptor::Clause(
      "at",
      {
        {51, {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "at", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_atomic_default_mem_order,
    descriptor::Clause(
      "atomic_default_mem_order",
      {
        {50, {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "atomic_default_mem_order", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_bind,
    descriptor::Clause(
      "bind",
      {
        {50, {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "bind", {Directive::OMPD_loop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_capture,
    descriptor::Clause(
      "capture",
      {
        {45, {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "capture", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_collapse,
    descriptor::Clause(
      "collapse",
      {
        {45, {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::OnceForAllConstituents, Property::Unique}}, "collapse", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_collector,
    descriptor::Clause(
      "collector",
      {
        {60, {{{Property::Required, Property::Unique}}, "collector", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required, Property::Unique}}, "collector", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_combiner,
    descriptor::Clause(
      "combiner",
      {
        {60, {{{Property::Required, Property::Unique}}, "combiner", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required, Property::Unique}}, "combiner", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_compare,
    descriptor::Clause(
      "compare",
      {
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "compare", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_contains,
    descriptor::Clause(
      "contains",
      {
        {51, {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "contains", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_copyin,
    descriptor::Clause(
      "copyin",
      {
        {45, {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataCopying, Property::OutermostLeaf}}, "copyin", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_copyprivate,
    descriptor::Clause(
      "copyprivate",
      {
        {45, {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataCopying, Property::EndClause, Property::InnermostLeaf}}, "copyprivate", {Directive::OMPD_single}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_counts,
    descriptor::Clause(
      "counts",
      {
        {60, {{{Property::Required, Property::Unique}}, "counts", {Directive::OMPD_split}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required, Property::Unique}}, "counts", {Directive::OMPD_split}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_default,
    descriptor::Clause(
      "default",
      {
        {45, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {51, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {52, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {60, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "default", {Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::VariableCategory}}},
        {61, {{{Property::DefaultAttribute, Property::PostModified}}, "default", {Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_default_variant,
    descriptor::Clause(
      "default-variant",
      {
        {50, {{{Property::Ultimate, Property::Unique}}, "default-variant", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Ultimate, Property::Unique}}, "default-variant", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
      }
    ),
  },
  {
    Clause::OMPC_defaultmap,
    descriptor::Clause(
      "defaultmap",
      {
        {45, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {50, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {51, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {52, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::VariableCategory}}},
        {60, {{{Property::DefaultAttribute, Property::PostModified, Property::Unique}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::VariableCategory}}},
        {61, {{{Property::DefaultAttribute, Property::PostModified}}, "defaultmap", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_depend,
    descriptor::Clause(
      "depend",
      {
        {45, {{{}}, "depend", {Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::TaskDependenceType}}},
        {50, {{{}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::Iterator, Modifier::TaskDependenceType}}},
        {51, {{{Property::DispatchRequirement}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_ordered_standalone, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::Iterator, Modifier::TaskDependenceType}}},
        {52, {{{Property::DispatchRequirement}}, "depend", {Directive::OMPD_depobj, Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::TaskDependenceType}}},
        {60, {{{Property::DispatchRequirement, Property::TaskInherited, Property::TaskgraphAltering}}, "depend", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::TaskDependenceType}}},
        {61, {{{Property::DispatchRequirement, Property::TaskInherited, Property::TaskSynchronizing, Property::TaskgraphAltering}}, "depend", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::TaskDependenceType}}},
      }
    ),
  },
  {
    Clause::OMPC_depth,
    descriptor::Clause(
      "depth",
      {
        {61, {{{Property::Unique}}, "depth", {Directive::OMPD_flatten, Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_destroy,
    descriptor::Clause(
      "destroy",
      {
        {50, {{{}}, "destroy", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{}}, "destroy", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_detach,
    descriptor::Clause(
      "detach",
      {
        {50, {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization, Property::Unique}}, "detach", {Directive::OMPD_target_data, Directive::OMPD_task}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_device,
    descriptor::Clause(
      "device",
      {
        {45, {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}}},
        {51, {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}}},
        {52, {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier}}},
        {60, {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier, Modifier::DirectiveNameModifier}}},
        {61, {{{Property::IcvDefaulted, Property::Unique}}, "device", {Directive::OMPD_dispatch, Directive::OMPD_interop, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DeviceModifier, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_device_safesync,
    descriptor::Clause(
      "device_safesync",
      {
        {60, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "device_safesync", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "device_safesync", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_device_type,
    descriptor::Clause(
      "device_type",
      {
        {50, {{{Property::Unique}}, "device_type", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target, Directive::OMPD_groupprivate, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "device_type", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target, Directive::OMPD_groupprivate, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_dist_schedule,
    descriptor::Clause(
      "dist_schedule",
      {
        {45, {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::ScheduleSpecification, Property::Unique}}, "dist_schedule", {Directive::OMPD_distribute}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_doacross,
    descriptor::Clause(
      "doacross",
      {
        {52, {{{Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType}}},
        {60, {{{Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required}}, "doacross", {Directive::OMPD_ordered_standalone}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DependenceType, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_dyn_groupprivate,
    descriptor::Clause(
      "dyn_groupprivate",
      {
        {61, {{{Property::TargetConsistent}}, "dyn_groupprivate", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AccessGroup, Modifier::DirectiveNameModifier, Modifier::FallbackModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_dynamic_allocators,
    descriptor::Clause(
      "dynamic_allocators",
      {
        {50, {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "dynamic_allocators", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_enter,
    descriptor::Clause(
      "enter",
      {
        {52, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AutomapModifier, Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "enter", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AutomapModifier, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_exclusive,
    descriptor::Clause(
      "exclusive",
      {
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "exclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_fail,
    descriptor::Clause(
      "fail",
      {
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "fail", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_filter,
    descriptor::Clause(
      "filter",
      {
        {51, {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "filter", {Directive::OMPD_masked}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_final,
    descriptor::Clause(
      "final",
      {
        {45, {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "final", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_firstprivate,
    descriptor::Clause(
      "firstprivate",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Saved}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization}}, "firstprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Saved}}},
      }
    ),
  },
  {
    Clause::OMPC_from,
    descriptor::Clause(
      "from",
      {
        {45, {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Mapper}}},
        {51, {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::Mapper, Modifier::MotionModifier}}},
        {52, {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Expectation, Modifier::Iterator, Modifier::Mapper}}},
        {60, {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
        {61, {{{Property::DataMotionAttribute}}, "from", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_full,
    descriptor::Clause(
      "full",
      {
        {51, {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "full", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_grainsize,
    descriptor::Clause(
      "grainsize",
      {
        {45, {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {52, {{{Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {60, {{{Property::TaskgraphAltering, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
        {61, {{{Property::TaskgraphAltering, Property::Unique}}, "grainsize", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
      }
    ),
  },
  {
    Clause::OMPC_graph_id,
    descriptor::Clause(
      "graph_id",
      {
        {60, {{{Property::Unique}}, "graph_id", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "graph_id", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_graph_reset,
    descriptor::Clause(
      "graph_reset",
      {
        {60, {{{Property::Unique}}, "graph_reset", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "graph_reset", {Directive::OMPD_taskgraph}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_has_device_addr,
    descriptor::Clause(
      "has_device_addr",
      {
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::OutermostLeaf}}, "has_device_addr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_hint,
    descriptor::Clause(
      "hint",
      {
        {45, {{{Property::Unique}}, "hint", {Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "hint", {Directive::OMPD_atomic, Directive::OMPD_critical}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_holds,
    descriptor::Clause(
      "holds",
      {
        {51, {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "holds", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_if,
    descriptor::Clause(
      "if",
      {
        {45, {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {50, {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {51, {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {52, {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {60, {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::TargetConsistent}}, "if", {Directive::OMPD_cancel, Directive::OMPD_parallel, Directive::OMPD_simd, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_in_reduction,
    descriptor::Clause(
      "in_reduction",
      {
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::Privatization, Property::ReductionParticipating}}, "in_reduction", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
      }
    ),
  },
  {
    Clause::OMPC_inbranch,
    descriptor::Clause(
      "inbranch",
      {
        {45, {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "inbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_inclusive,
    descriptor::Clause(
      "inclusive",
      {
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "inclusive", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_indirect,
    descriptor::Clause(
      "indirect",
      {
        {51, {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "indirect", {Directive::OMPD_begin_declare_target, Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_induction,
    descriptor::Clause(
      "induction",
      {
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "induction", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::InductionIdentifier, Modifier::InductionModifier, Modifier::StepModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "induction", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::InductionIdentifier, Modifier::InductionModifier, Modifier::StepModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_inductor,
    descriptor::Clause(
      "inductor",
      {
        {60, {{{Property::Required, Property::Unique}}, "inductor", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required, Property::Unique}}, "inductor", {Directive::OMPD_declare_induction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_init,
    descriptor::Clause(
      "init",
      {
        {51, {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::InteropType}}},
        {52, {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::InteropType}}},
        {60, {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DepinfoModifier, Modifier::DirectiveNameModifier, Modifier::InteropType, Modifier::PreferType}}},
        {61, {{{Property::InnermostLeaf}}, "init", {Directive::OMPD_depobj, Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DepinfoModifier, Modifier::DirectiveNameModifier, Modifier::PreferType}}},
      }
    ),
  },
  {
    Clause::OMPC_init_complete,
    descriptor::Clause(
      "init_complete",
      {
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "init_complete", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "init_complete", {Directive::OMPD_scan}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_initializer,
    descriptor::Clause(
      "initializer",
      {
        {50, {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "initializer", {Directive::OMPD_declare_reduction}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_interop,
    descriptor::Clause(
      "interop",
      {
        {60, {{{Property::DispatchRequirement, Property::Unique}}, "interop", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DispatchRequirement, Property::Unique}}, "interop", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_is_device_ptr,
    descriptor::Clause(
      "is_device_ptr",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::DispatchRequirement, Property::InnermostLeaf, Property::Privatization}}, "is_device_ptr", {Directive::OMPD_dispatch, Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_lastprivate,
    descriptor::Clause(
      "lastprivate",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LastprivateModifier}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LastprivateModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization}}, "lastprivate", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LastprivateModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_linear,
    descriptor::Clause(
      "linear",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::LinearStep}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::PostModified, Property::Privatization}}, "linear", {Directive::OMPD_declare_simd, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LinearModifier, Modifier::StepComplexModifier, Modifier::StepSimpleModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_link,
    descriptor::Clause(
      "link",
      {
        {45, {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute}}, "link", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_local,
    descriptor::Clause(
      "local",
      {
        {60, {{{Property::DataEnvironmentAttribute}}, "local", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute}}, "local", {Directive::OMPD_declare_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_looprange,
    descriptor::Clause(
      "looprange",
      {
        {60, {{{Property::Unique}}, "looprange", {Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "looprange", {Directive::OMPD_fuse}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_map,
    descriptor::Clause(
      "map",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MapType, Modifier::MapTypeModifier, Modifier::OmpxHoldModifier}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::MapType, Modifier::MapTypeModifier, Modifier::Mapper, Modifier::OmpxHoldModifier}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AlwaysModifier, Modifier::CloseModifier, Modifier::DeleteModifier, Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::MapType, Modifier::Mapper, Modifier::OmpxHoldModifier, Modifier::PresentModifier, Modifier::RefModifier, Modifier::SelfModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataMappingAttribute}}, "map", {Directive::OMPD_declare_mapper, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::AttachModifier, Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::MapType, Modifier::Mapper, Modifier::OmpxHoldModifier, Modifier::RefModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_match,
    descriptor::Clause(
      "match",
      {
        {50, {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required, Property::Unique}}, "match", {Directive::OMPD_begin_declare_variant, Directive::OMPD_declare_variant}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_memscope,
    descriptor::Clause(
      "memscope",
      {
        {60, {{{Property::Unique}}, "memscope", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "memscope", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_mergeable,
    descriptor::Clause(
      "mergeable",
      {
        {45, {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "mergeable", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "mergeable", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "mergeable", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_message,
    descriptor::Clause(
      "message",
      {
        {51, {{{Property::Unique}}, "message", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "message", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "message", {Directive::OMPD_error, Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "message", {Directive::OMPD_error, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp,
    descriptor::Clause(
      "no_openmp",
      {
        {51, {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "no_openmp", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp_constructs,
    descriptor::Clause(
      "no_openmp_constructs",
      {
        {60, {{{Property::Unique}}, "no_openmp_constructs", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "no_openmp_constructs", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_openmp_routines,
    descriptor::Clause(
      "no_openmp_routines",
      {
        {51, {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "no_openmp_routines", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_no_parallelism,
    descriptor::Clause(
      "no_parallelism",
      {
        {51, {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "no_parallelism", {Directive::OMPD_assume, Directive::OMPD_assumes, Directive::OMPD_begin_assumes}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nocontext,
    descriptor::Clause(
      "nocontext",
      {
        {51, {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "nocontext", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nogroup,
    descriptor::Clause(
      "nogroup",
      {
        {45, {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::OutermostLeaf, Property::Unique}}, "nogroup", {Directive::OMPD_target_data, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::OutermostLeaf, Property::TaskSynchronizing, Property::Unique}}, "nogroup", {Directive::OMPD_target_data, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nontemporal,
    descriptor::Clause(
      "nontemporal",
      {
        {50, {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{}}, "nontemporal", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_notinbranch,
    descriptor::Clause(
      "notinbranch",
      {
        {45, {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "notinbranch", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_novariants,
    descriptor::Clause(
      "novariants",
      {
        {51, {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "novariants", {Directive::OMPD_dispatch}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_nowait,
    descriptor::Clause(
      "nowait",
      {
        {45, {{{Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DispatchRequirement, Property::EndClause, Property::OutermostLeaf, Property::TaskSynchronizing, Property::Unique}}, "nowait", {Directive::OMPD_dispatch, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_interop, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_taskwait, Directive::OMPD_workshare}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_num_tasks,
    descriptor::Clause(
      "num_tasks",
      {
        {45, {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {52, {{{Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Prescriptiveness}}},
        {60, {{{Property::TaskgraphAltering, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
        {61, {{{Property::TaskgraphAltering, Property::Unique}}, "num_tasks", {Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
      }
    ),
  },
  {
    Clause::OMPC_num_teams,
    descriptor::Clause(
      "num_teams",
      {
        {45, {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}}},
        {51, {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}}},
        {52, {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::LowerBound}}},
        {60, {{{Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::LowerBound}}},
        {61, {{{Property::SpaceConfiguring, Property::TargetConsistent, Property::Unique}}, "num_teams", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::LowerBound}}},
      }
    ),
  },
  {
    Clause::OMPC_num_threads,
    descriptor::Clause(
      "num_threads",
      {
        {45, {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Prescriptiveness}}},
        {61, {{{Property::SpaceConfiguring, Property::Unique}}, "num_threads", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::PrescriptivenessModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_order,
    descriptor::Clause(
      "order",
      {
        {50, {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::OrderModifier}}},
        {52, {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::OrderModifier}}},
        {60, {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OrderModifier}}},
        {61, {{{Property::ScheduleSpecification, Property::Unique}}, "order", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OrderModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_ordered,
    descriptor::Clause(
      "ordered",
      {
        {45, {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::OnceForAllConstituents, Property::Unique}}, "ordered", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_otherwise,
    descriptor::Clause(
      "otherwise",
      {
        {52, {{{Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Ultimate, Property::Unique}}, "otherwise", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_partial,
    descriptor::Clause(
      "partial",
      {
        {51, {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "partial", {Directive::OMPD_unroll}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_permutation,
    descriptor::Clause(
      "permutation",
      {
        {60, {{{Property::Unique}}, "permutation", {Directive::OMPD_interchange}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "permutation", {Directive::OMPD_interchange}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_priority,
    descriptor::Clause(
      "priority",
      {
        {45, {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "priority", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "priority", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "priority", {Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskgraph, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_private,
    descriptor::Clause(
      "private",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::InnermostLeaf, Property::Privatization}}, "private", {Directive::OMPD_distribute, Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_single, Directive::OMPD_target, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_proc_bind,
    descriptor::Clause(
      "proc_bind",
      {
        {45, {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "proc_bind", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_read,
    descriptor::Clause(
      "read",
      {
        {45, {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "read", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_reduction,
    descriptor::Clause(
      "reduction",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OriginalSharingModifier, Modifier::ReductionIdentifier, Modifier::ReductionModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionParticipating, Property::ReductionScoping}}, "reduction", {Directive::OMPD_do, Directive::OMPD_for, Directive::OMPD_loop, Directive::OMPD_parallel, Directive::OMPD_scope, Directive::OMPD_sections, Directive::OMPD_simd, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::OriginalSharingModifier, Modifier::PerThreadModifier, Modifier::ReductionIdentifier}}},
      }
    ),
  },
  {
    Clause::OMPC_relaxed,
    descriptor::Clause(
      "relaxed",
      {
        {50, {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "relaxed", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_release,
    descriptor::Clause(
      "release",
      {
        {50, {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "release", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_replayable,
    descriptor::Clause(
      "replayable",
      {
        {60, {{{Property::Unique}}, "replayable", {Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "replayable", {Directive::OMPD_target, Directive::OMPD_target_enter_data, Directive::OMPD_target_exit_data, Directive::OMPD_target_update, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_taskwait}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_reverse_offload,
    descriptor::Clause(
      "reverse_offload",
      {
        {50, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "reverse_offload", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_safelen,
    descriptor::Clause(
      "safelen",
      {
        {45, {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "safelen", {Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_safesync,
    descriptor::Clause(
      "safesync",
      {
        {60, {{{Property::Unique}}, "safesync", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "safesync", {Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_schedule,
    descriptor::Clause(
      "schedule",
      {
        {45, {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {50, {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {51, {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {52, {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::OrderingModifier}}},
        {60, {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::DirectiveNameModifier, Modifier::OrderingModifier}}},
        {61, {{{Property::ScheduleSpecification, Property::Unique}}, "schedule", {Directive::OMPD_do, Directive::OMPD_for}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ChunkModifier, Modifier::DirectiveNameModifier, Modifier::OrderingModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_self_maps,
    descriptor::Clause(
      "self_maps",
      {
        {60, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "self_maps", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "self_maps", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_seq_cst,
    descriptor::Clause(
      "seq_cst",
      {
        {45, {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "seq_cst", {Directive::OMPD_atomic, Directive::OMPD_flush}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_severity,
    descriptor::Clause(
      "severity",
      {
        {51, {{{Property::Unique}}, "severity", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "severity", {Directive::OMPD_error}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "severity", {Directive::OMPD_error, Directive::OMPD_parallel}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "severity", {Directive::OMPD_error, Directive::OMPD_parallel, Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_shared,
    descriptor::Clause(
      "shared",
      {
        {45, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "shared", {Directive::OMPD_parallel, Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_simd,
    descriptor::Clause(
      "simd",
      {
        {45, {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "simd", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_simdlen,
    descriptor::Clause(
      "simdlen",
      {
        {45, {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "simdlen", {Directive::OMPD_declare_simd, Directive::OMPD_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ScaledModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_sizes,
    descriptor::Clause(
      "sizes",
      {
        {51, {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_stripe, Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Required, Property::Unique}}, "sizes", {Directive::OMPD_stripe, Directive::OMPD_tile}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::SizesSelector}}},
      }
    ),
  },
  {
    Clause::OMPC_task_reduction,
    descriptor::Clause(
      "task_reduction",
      {
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ReductionIdentifier}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::OriginalListItemUpdating, Property::Privatization, Property::ReductionScoping}}, "task_reduction", {Directive::OMPD_taskgroup}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::ReductionIdentifier}}},
      }
    ),
  },
  {
    Clause::OMPC_thread_limit,
    descriptor::Clause(
      "thread_limit",
      {
        {45, {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::IcvModifying, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::IcvModifying, Property::SpaceConfiguring, Property::TargetConsistent, Property::Unique}}, "thread_limit", {Directive::OMPD_target, Directive::OMPD_teams}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DimsModifier, Modifier::DirectiveNameModifier, Modifier::PrescriptivenessModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_threads,
    descriptor::Clause(
      "threads",
      {
        {45, {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "threads", {Directive::OMPD_ordered_blockassoc}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_threadset,
    descriptor::Clause(
      "threadset",
      {
        {60, {{{Property::Unique}}, "threadset", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "threadset", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_to,
    descriptor::Clause(
      "to",
      {
        {45, {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Mapper}}},
        {51, {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_declare_target, Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Iterator, Modifier::Mapper, Modifier::MotionModifier}}},
        {52, {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::Expectation, Modifier::Iterator, Modifier::Mapper}}},
        {60, {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
        {61, {{{Property::DataMotionAttribute}}, "to", {Directive::OMPD_target_update}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Iterator, Modifier::Mapper, Modifier::PresentModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_transparent,
    descriptor::Clause(
      "transparent",
      {
        {60, {{{Property::Unique}}, "transparent", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "transparent", {Directive::OMPD_target_data, Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_unified_address,
    descriptor::Clause(
      "unified_address",
      {
        {50, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_address", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_unified_shared_memory,
    descriptor::Clause(
      "unified_shared_memory",
      {
        {50, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DeviceGlobalRequirement, Property::Unique}}, "unified_shared_memory", {Directive::OMPD_requires}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_uniform,
    descriptor::Clause(
      "uniform",
      {
        {45, {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::DataEnvironmentAttribute}}, "uniform", {Directive::OMPD_declare_simd}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_untied,
    descriptor::Clause(
      "untied",
      {
        {45, {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::Unique}}, "untied", {Directive::OMPD_task, Directive::OMPD_taskloop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_update,
    descriptor::Clause(
      "update",
      {
        {45, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_update_depend_objects,
    descriptor::Clause(
      "update-depend_objects",
      {
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::TaskDependenceType}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "update", {Directive::OMPD_depobj}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::TaskDependenceType}}},
      }
    ),
  },
  {
    Clause::OMPC_use,
    descriptor::Clause(
      "use",
      {
        {51, {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{}}, "use", {Directive::OMPD_interop}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_use_device_addr,
    descriptor::Clause(
      "use_device_addr",
      {
        {50, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated}}, "use_device_addr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_use_device_ptr,
    descriptor::Clause(
      "use_device_ptr",
      {
        {45, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::AllDataEnvironments, Property::DataEnvironmentAttribute, Property::DataSharingAttribute, Property::DeviceAssociated, Property::Privatization}}, "use_device_ptr", {Directive::OMPD_target_data}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::Fallback}}},
      }
    ),
  },
  {
    Clause::OMPC_uses_allocators,
    descriptor::Clause(
      "uses_allocators",
      {
        {50, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::MemSpace, Modifier::TraitsArray}}},
        {60, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::MemSpace, Modifier::TraitsArray}}},
        {61, {{{Property::DataEnvironmentAttribute, Property::DataSharingAttribute}}, "uses_allocators", {Directive::OMPD_target}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier, Modifier::MemSpace, Modifier::TraitsArray}}},
      }
    ),
  },
  {
    Clause::OMPC_weak,
    descriptor::Clause(
      "weak",
      {
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "weak", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_when,
    descriptor::Clause(
      "when",
      {
        {50, {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}}},
        {51, {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}}},
        {52, {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector}}},
        {60, {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector, Modifier::DirectiveNameModifier}}},
        {61, {{{}}, "when", {Directive::OMPD_metadirective}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::ContextSelector, Modifier::DirectiveNameModifier}}},
      }
    ),
  },
  {
    Clause::OMPC_write,
    descriptor::Clause(
      "write",
      {
        {45, {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {50, {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {51, {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {52, {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {}}},
        {60, {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
        {61, {{{Property::InnermostLeaf, Property::Unique}}, "write", {Directive::OMPD_atomic}, SourceLanguage::C | SourceLanguage::Fortran, {Modifier::DirectiveNameModifier}}},
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
        {61, {{{Property::Unique}}, {Clause::OMPC_dyn_groupprivate}}},
      }
    ),
  },
  {
    Modifier::AdjustOp,
    descriptor::Modifier(
      "adjust-op",
      {
        {51, {{{Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
        {52, {{{Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
        {60, {{{Property::Required, Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::AlignModifier,
    descriptor::Modifier(
      "align-modifier",
      {
        {51, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::Alignment,
    descriptor::Modifier(
      "alignment",
      {
        {45, {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {50, {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {51, {{{Property::Positive, Property::RegionInvariant, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {52, {{{Property::Positive, Property::RegionInvariant, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {60, {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
        {61, {{{Property::Constant, Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_aligned}}},
      }
    ),
  },
  {
    Modifier::AllocatorComplexModifier,
    descriptor::Modifier(
      "allocator-complex-modifier",
      {
        {52, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::AllocatorSimpleModifier,
    descriptor::Modifier(
      "allocator-simple-modifier",
      {
        {50, {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {51, {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {52, {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {60, {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
        {61, {{{Property::Exclusive, Property::Unique}}, {Clause::OMPC_allocate}}},
      }
    ),
  },
  {
    Modifier::AlwaysModifier,
    descriptor::Modifier(
      "always-modifier",
      {
        {60, {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::AttachModifier,
    descriptor::Modifier(
      "attach-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::AutomapModifier,
    descriptor::Modifier(
      "automap-modifier",
      {
        {60, {{{Property::Unique}}, {Clause::OMPC_enter}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_enter}}},
      }
    ),
  },
  {
    Modifier::ChunkModifier,
    descriptor::Modifier(
      "chunk-modifier",
      {
        {45, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
      }
    ),
  },
  {
    Modifier::CloseModifier,
    descriptor::Modifier(
      "close-modifier",
      {
        {60, {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::ContextSelector,
    descriptor::Modifier(
      "context-selector",
      {
        {50, {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {51, {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {52, {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {60, {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
        {61, {{{Property::Required, Property::Unique}}, {Clause::OMPC_when}}},
      }
    ),
  },
  {
    Modifier::DefaultModifier,
    descriptor::Modifier(
      "default-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::DeleteModifier,
    descriptor::Modifier(
      "delete-modifier",
      {
        {60, {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::DependenceType,
    descriptor::Modifier(
      "dependence-type",
      {
        {45, {{{Property::Unique}}, {Clause::OMPC_depend}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_depend}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_depend}}},
        {52, {{{Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
        {60, {{{Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
        {61, {{{Property::Required, Property::Unique}}, {Clause::OMPC_doacross}}},
      }
    ),
  },
  {
    Modifier::DepinfoModifier,
    descriptor::Modifier(
      "depinfo-modifier",
      {
        {60, {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
        {61, {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::DepobjModifier,
    descriptor::Modifier(
      "depobj-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    Modifier::DeviceModifier,
    descriptor::Modifier(
      "device-modifier",
      {
        {50, {{{Property::Unique}}, {Clause::OMPC_device}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_device}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_device}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_device}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_device}}},
      }
    ),
  },
  {
    Modifier::DimsModifier,
    descriptor::Modifier(
      "dims-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_thread_limit}}},
      }
    ),
  },
  {
    Modifier::DirectiveNameModifier,
    descriptor::Modifier(
      "directive-name-modifier",
      {
        {45, {{{Property::Unique}}, {Clause::OMPC_if}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_if}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_if}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_if}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_adjust_args, Clause::OMPC_affinity, Clause::OMPC_align, Clause::OMPC_aligned, Clause::OMPC_allocate, Clause::OMPC_allocator, Clause::OMPC_append_args, Clause::OMPC_apply, Clause::OMPC_at, Clause::OMPC_atomic_default_mem_order, Clause::OMPC_bind, Clause::OMPC_capture, Clause::OMPC_collapse, Clause::OMPC_collector, Clause::OMPC_combiner, Clause::OMPC_compare, Clause::OMPC_contains, Clause::OMPC_copyin, Clause::OMPC_copyprivate, Clause::OMPC_counts, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_destroy, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_device_safesync, Clause::OMPC_device_type, Clause::OMPC_dist_schedule, Clause::OMPC_doacross, Clause::OMPC_dynamic_allocators, Clause::OMPC_enter, Clause::OMPC_exclusive, Clause::OMPC_fail, Clause::OMPC_filter, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_from, Clause::OMPC_full, Clause::OMPC_grainsize, Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_has_device_addr, Clause::OMPC_hint, Clause::OMPC_holds, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_inbranch, Clause::OMPC_inclusive, Clause::OMPC_indirect, Clause::OMPC_induction, Clause::OMPC_inductor, Clause::OMPC_init, Clause::OMPC_init_complete, Clause::OMPC_initializer, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_link, Clause::OMPC_local, Clause::OMPC_looprange, Clause::OMPC_map, Clause::OMPC_match, Clause::OMPC_memscope, Clause::OMPC_mergeable, Clause::OMPC_message, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism, Clause::OMPC_nocontext, Clause::OMPC_nogroup, Clause::OMPC_nontemporal, Clause::OMPC_notinbranch, Clause::OMPC_novariants, Clause::OMPC_nowait, Clause::OMPC_num_tasks, Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_otherwise, Clause::OMPC_partial, Clause::OMPC_permutation, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_read, Clause::OMPC_reduction, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_replayable, Clause::OMPC_reverse_offload, Clause::OMPC_safelen, Clause::OMPC_safesync, Clause::OMPC_schedule, Clause::OMPC_self_maps, Clause::OMPC_seq_cst, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_simd, Clause::OMPC_simdlen, Clause::OMPC_sizes, Clause::OMPC_task_reduction, Clause::OMPC_thread_limit, Clause::OMPC_threads, Clause::OMPC_threadset, Clause::OMPC_to, Clause::OMPC_transparent, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory, Clause::OMPC_uniform, Clause::OMPC_untied, Clause::OMPC_update, Clause::OMPC_update_depend_objects, Clause::OMPC_use, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr, Clause::OMPC_uses_allocators, Clause::OMPC_weak, Clause::OMPC_when, Clause::OMPC_write}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_absent, Clause::OMPC_acq_rel, Clause::OMPC_acquire, Clause::OMPC_adjust_args, Clause::OMPC_affinity, Clause::OMPC_align, Clause::OMPC_aligned, Clause::OMPC_allocate, Clause::OMPC_allocator, Clause::OMPC_append_args, Clause::OMPC_apply, Clause::OMPC_at, Clause::OMPC_atomic_default_mem_order, Clause::OMPC_bind, Clause::OMPC_capture, Clause::OMPC_collapse, Clause::OMPC_collector, Clause::OMPC_combiner, Clause::OMPC_compare, Clause::OMPC_contains, Clause::OMPC_copyin, Clause::OMPC_copyprivate, Clause::OMPC_counts, Clause::OMPC_default, Clause::OMPC_defaultmap, Clause::OMPC_depend, Clause::OMPC_depth, Clause::OMPC_destroy, Clause::OMPC_detach, Clause::OMPC_device, Clause::OMPC_device_safesync, Clause::OMPC_device_type, Clause::OMPC_dist_schedule, Clause::OMPC_doacross, Clause::OMPC_dyn_groupprivate, Clause::OMPC_dynamic_allocators, Clause::OMPC_enter, Clause::OMPC_exclusive, Clause::OMPC_fail, Clause::OMPC_filter, Clause::OMPC_final, Clause::OMPC_firstprivate, Clause::OMPC_from, Clause::OMPC_full, Clause::OMPC_grainsize, Clause::OMPC_graph_id, Clause::OMPC_graph_reset, Clause::OMPC_has_device_addr, Clause::OMPC_hint, Clause::OMPC_holds, Clause::OMPC_if, Clause::OMPC_in_reduction, Clause::OMPC_inbranch, Clause::OMPC_inclusive, Clause::OMPC_indirect, Clause::OMPC_induction, Clause::OMPC_inductor, Clause::OMPC_init, Clause::OMPC_init_complete, Clause::OMPC_initializer, Clause::OMPC_interop, Clause::OMPC_is_device_ptr, Clause::OMPC_lastprivate, Clause::OMPC_linear, Clause::OMPC_link, Clause::OMPC_local, Clause::OMPC_looprange, Clause::OMPC_map, Clause::OMPC_match, Clause::OMPC_memscope, Clause::OMPC_mergeable, Clause::OMPC_message, Clause::OMPC_no_openmp, Clause::OMPC_no_openmp_constructs, Clause::OMPC_no_openmp_routines, Clause::OMPC_no_parallelism, Clause::OMPC_nocontext, Clause::OMPC_nogroup, Clause::OMPC_nontemporal, Clause::OMPC_notinbranch, Clause::OMPC_novariants, Clause::OMPC_nowait, Clause::OMPC_num_tasks, Clause::OMPC_num_teams, Clause::OMPC_num_threads, Clause::OMPC_order, Clause::OMPC_ordered, Clause::OMPC_otherwise, Clause::OMPC_partial, Clause::OMPC_permutation, Clause::OMPC_priority, Clause::OMPC_private, Clause::OMPC_proc_bind, Clause::OMPC_read, Clause::OMPC_reduction, Clause::OMPC_relaxed, Clause::OMPC_release, Clause::OMPC_replayable, Clause::OMPC_reverse_offload, Clause::OMPC_safelen, Clause::OMPC_safesync, Clause::OMPC_schedule, Clause::OMPC_self_maps, Clause::OMPC_seq_cst, Clause::OMPC_severity, Clause::OMPC_shared, Clause::OMPC_simd, Clause::OMPC_simdlen, Clause::OMPC_sizes, Clause::OMPC_task_reduction, Clause::OMPC_thread_limit, Clause::OMPC_threads, Clause::OMPC_threadset, Clause::OMPC_to, Clause::OMPC_transparent, Clause::OMPC_unified_address, Clause::OMPC_unified_shared_memory, Clause::OMPC_uniform, Clause::OMPC_untied, Clause::OMPC_update, Clause::OMPC_update_depend_objects, Clause::OMPC_use, Clause::OMPC_use_device_addr, Clause::OMPC_use_device_ptr, Clause::OMPC_uses_allocators, Clause::OMPC_weak, Clause::OMPC_when, Clause::OMPC_write}}},
      }
    ),
  },
  {
    Modifier::Expectation,
    descriptor::Modifier(
      "expectation",
      {
        {52, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::Fallback,
    descriptor::Modifier(
      "fallback",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_use_device_ptr}}},
      }
    ),
  },
  {
    Modifier::FallbackModifier,
    descriptor::Modifier(
      "fallback-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_dyn_groupprivate}}},
      }
    ),
  },
  {
    Modifier::InductionIdentifier,
    descriptor::Modifier(
      "induction-identifier",
      {
        {60, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_induction}}},
        {61, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::InductionModifier,
    descriptor::Modifier(
      "induction-modifier",
      {
        {60, {{{Property::Unique}}, {Clause::OMPC_induction}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::InscanModifier,
    descriptor::Modifier(
      "inscan-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::InteropType,
    descriptor::Modifier(
      "interop-type",
      {
        {51, {{{Property::Repeatable, Property::Required}}, {Clause::OMPC_init}}},
        {52, {{{Property::Repeatable, Property::Required}}, {Clause::OMPC_init}}},
        {60, {{{Property::Repeatable}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::Iterator,
    descriptor::Modifier(
      "iterator",
      {
        {50, {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::LastprivateModifier,
    descriptor::Modifier(
      "lastprivate-modifier",
      {
        {50, {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_lastprivate}}},
      }
    ),
  },
  {
    Modifier::LinearModifier,
    descriptor::Modifier(
      "linear-modifier",
      {
        {45, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::LinearStep,
    descriptor::Modifier(
      "linear-step",
      {
        {45, {{{Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {50, {{{Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {51, {{{Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::LocModifier,
    descriptor::Modifier(
      "loc-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_depend}}},
      }
    ),
  },
  {
    Modifier::LoopModifier,
    descriptor::Modifier(
      "loop-modifier",
      {
        {60, {{{Property::Optional, Property::Unique}}, {Clause::OMPC_apply}}},
        {61, {{{Property::Optional, Property::Unique}}, {Clause::OMPC_apply}}},
      }
    ),
  },
  {
    Modifier::LowerBound,
    descriptor::Modifier(
      "lower-bound",
      {
        {50, {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {51, {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {52, {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {60, {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
        {61, {{{Property::Positive, Property::Ultimate, Property::Unique}}, {Clause::OMPC_num_teams}}},
      }
    ),
  },
  {
    Modifier::MapType,
    descriptor::Modifier(
      "map-type",
      {
        {45, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {50, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {51, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {52, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_map}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::MapTypeModifier,
    descriptor::Modifier(
      "map-type-modifier",
      {
        {45, {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
        {50, {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
        {51, {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
        {52, {{{Property::MapTypeModifying, Property::Repeatable}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::Mapper,
    descriptor::Modifier(
      "mapper",
      {
        {50, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::MemSpace,
    descriptor::Modifier(
      "mem-space",
      {
        {52, {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
      }
    ),
  },
  {
    Modifier::MotionModifier,
    descriptor::Modifier(
      "motion-modifier",
      {
        {51, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::NeedDeviceAddr,
    descriptor::Modifier(
      "need-device-addr",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::NeedDevicePtr,
    descriptor::Modifier(
      "need-device-ptr",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::NoncommutativeModifier,
    descriptor::Modifier(
      "noncommutative-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::Nothing,
    descriptor::Modifier(
      "nothing",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_adjust_args}}},
      }
    ),
  },
  {
    Modifier::OmpxHoldModifier,
    descriptor::Modifier(
      "ompx-hold-modifier",
      {
        {45, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::OptionalModifier,
    descriptor::Modifier(
      "optional-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
  {
    Modifier::OrderModifier,
    descriptor::Modifier(
      "order-modifier",
      {
        {51, {{{Property::Unique}}, {Clause::OMPC_order}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_order}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_order}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_order}}},
      }
    ),
  },
  {
    Modifier::OrderingModifier,
    descriptor::Modifier(
      "ordering-modifier",
      {
        {45, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_schedule}}},
      }
    ),
  },
  {
    Modifier::OriginalSharingModifier,
    descriptor::Modifier(
      "original-sharing-modifier",
      {
        {60, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::PerThreadModifier,
    descriptor::Modifier(
      "per-thread-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::PreferType,
    descriptor::Modifier(
      "prefer-type",
      {
        {60, {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
        {61, {{{Property::Complex, Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::Prescriptiveness,
    descriptor::Modifier(
      "prescriptiveness",
      {
        {51, {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks, Clause::OMPC_num_threads}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}}},
      }
    ),
  },
  {
    Modifier::PrescriptivenessModifier,
    descriptor::Modifier(
      "prescriptiveness-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_num_threads, Clause::OMPC_thread_limit}}},
      }
    ),
  },
  {
    Modifier::PresentModifier,
    descriptor::Modifier(
      "present-modifier",
      {
        {60, {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}}},
      }
    ),
  },
  {
    Modifier::ReductionIdentifier,
    descriptor::Modifier(
      "reduction-identifier",
      {
        {45, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_reduction}}},
        {50, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {51, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {52, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {60, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
        {61, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_in_reduction, Clause::OMPC_reduction, Clause::OMPC_task_reduction}}},
      }
    ),
  },
  {
    Modifier::ReductionModifier,
    descriptor::Modifier(
      "reduction-modifier",
      {
        {50, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::RefModifier,
    descriptor::Modifier(
      "ref-modifier",
      {
        {60, {{{Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::Saved,
    descriptor::Modifier(
      "saved",
      {
        {60, {{{Property::Unique}}, {Clause::OMPC_firstprivate}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_firstprivate}}},
      }
    ),
  },
  {
    Modifier::ScaledModifier,
    descriptor::Modifier(
      "scaled-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_simdlen}}},
      }
    ),
  },
  {
    Modifier::SelfModifier,
    descriptor::Modifier(
      "self-modifier",
      {
        {60, {{{Property::MapTypeModifying, Property::Unique}}, {Clause::OMPC_map}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_map}}},
      }
    ),
  },
  {
    Modifier::SizesSelector,
    descriptor::Modifier(
      "sizes-selector",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_sizes}}},
      }
    ),
  },
  {
    Modifier::StepComplexModifier,
    descriptor::Modifier(
      "step-complex-modifier",
      {
        {52, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_linear}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::StepModifier,
    descriptor::Modifier(
      "step-modifier",
      {
        {52, {{{Property::Required, Property::Unique}}, {}}},
        {60, {{{Property::Required, Property::Unique}}, {Clause::OMPC_induction}}},
        {61, {{{Property::Required, Property::Unique}}, {Clause::OMPC_induction}}},
      }
    ),
  },
  {
    Modifier::StepSimpleModifier,
    descriptor::Modifier(
      "step-simple-modifier",
      {
        {52, {{{Property::Exclusive, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {60, {{{Property::Exclusive, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
        {61, {{{Property::Exclusive, Property::RegionInvariant, Property::Unique}}, {Clause::OMPC_linear}}},
      }
    ),
  },
  {
    Modifier::TargetType,
    descriptor::Modifier(
      "target-type",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::TargetsyncType,
    descriptor::Modifier(
      "targetsync-type",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_init}}},
      }
    ),
  },
  {
    Modifier::TaskDependenceType,
    descriptor::Modifier(
      "task-dependence-type",
      {
        {45, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {50, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {51, {{{Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {52, {{{Property::Required, Property::Ultimate, Property::Unique}}, {Clause::OMPC_depend}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_update_depend_objects}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_depend, Clause::OMPC_update_depend_objects}}},
      }
    ),
  },
  {
    Modifier::TaskModifier,
    descriptor::Modifier(
      "task-modifier",
      {
        {61, {{{Property::Unique}}, {Clause::OMPC_reduction}}},
      }
    ),
  },
  {
    Modifier::TraitsArray,
    descriptor::Modifier(
      "traits-array",
      {
        {52, {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_uses_allocators}}},
      }
    ),
  },
  {
    Modifier::VariableCategory,
    descriptor::Modifier(
      "variable-category",
      {
        {45, {{{Property::Required, Property::Unique}}, {Clause::OMPC_defaultmap}}},
        {50, {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {51, {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {52, {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {60, {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
        {61, {{{Property::Unique}}, {Clause::OMPC_default, Clause::OMPC_defaultmap}}},
      }
    ),
  },
#endif // GEN_OMP_MODIFIER_DESCRIPTORS
    // clang-format on
