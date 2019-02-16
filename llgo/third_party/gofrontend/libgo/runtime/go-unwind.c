/* go-unwind.c -- unwind the stack for panic/recover.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"

#include <stdlib.h>
#include <unistd.h>

#include "unwind.h"
#define NO_SIZE_OF_ENCODED_VALUE
#include "unwind-pe.h"

#include "runtime.h"
#include "go-alloc.h"
#include "go-defer.h"
#include "go-panic.h"

/* The code for a Go exception.  */

#ifdef __ARM_EABI_UNWINDER__
static const _Unwind_Exception_Class __go_exception_class =
  { 'G', 'N', 'U', 'C', 'G', 'O', '\0', '\0' };
#else
static const _Unwind_Exception_Class __go_exception_class =
  ((((((((_Unwind_Exception_Class) 'G' 
         << 8 | (_Unwind_Exception_Class) 'N')
        << 8 | (_Unwind_Exception_Class) 'U')
       << 8 | (_Unwind_Exception_Class) 'C')
      << 8 | (_Unwind_Exception_Class) 'G')
     << 8 | (_Unwind_Exception_Class) 'O')
    << 8 | (_Unwind_Exception_Class) '\0')
   << 8 | (_Unwind_Exception_Class) '\0');
#endif


/* This function is called by exception handlers used when unwinding
   the stack after a recovered panic.  The exception handler looks
   like this:
     __go_check_defer (frame);
     return;
   If we have not yet reached the frame we are looking for, we
   continue unwinding.  */

void
__go_check_defer (_Bool *frame)
{
  G *g;
  struct _Unwind_Exception *hdr;

  g = runtime_g ();

  if (g == NULL)
    {
      /* Some other language has thrown an exception.  We know there
	 are no defer handlers, so there is nothing to do.  */
    }
  else if (g->is_foreign)
    {
      struct __go_panic_stack *n;
      _Bool was_recovered;

      /* Some other language has thrown an exception.  We need to run
	 the local defer handlers.  If they call recover, we stop
	 unwinding the stack here.  */

      n = ((struct __go_panic_stack *)
	   __go_alloc (sizeof (struct __go_panic_stack)));

      n->__arg.__type_descriptor = NULL;
      n->__arg.__object = NULL;
      n->__was_recovered = 0;
      n->__is_foreign = 1;
      n->__next = g->panic;
      g->panic = n;

      while (1)
	{
	  struct __go_defer_stack *d;
	  void (*pfn) (void *);

	  d = g->defer;
	  if (d == NULL || d->__frame != frame || d->__pfn == NULL)
	    break;

	  pfn = d->__pfn;
	  g->defer = d->__next;

	  (*pfn) (d->__arg);

	  if (runtime_m () != NULL)
	    runtime_freedefer (d);

	  if (n->__was_recovered)
	    {
	      /* The recover function caught the panic thrown by some
		 other language.  */
	      break;
	    }
	}

      was_recovered = n->__was_recovered;
      g->panic = n->__next;
      __go_free (n);

      if (was_recovered)
	{
	  /* Just return and continue executing Go code.  */
	  *frame = 1;
	  return;
	}

      /* We are panicing through this function.  */
      *frame = 0;
    }
  else if (g->defer != NULL
	   && g->defer->__pfn == NULL
	   && g->defer->__frame == frame)
    {
      struct __go_defer_stack *d;

      /* This is the defer function which called recover.  Simply
	 return to stop the stack unwind, and let the Go code continue
	 to execute.  */
      d = g->defer;
      g->defer = d->__next;

      if (runtime_m () != NULL)
	runtime_freedefer (d);

      /* We are returning from this function.  */
      *frame = 1;

      return;
    }

  /* This is some other defer function.  It was already run by the
     call to panic, or just above.  Rethrow the exception.  */

  hdr = (struct _Unwind_Exception *) g->exception;

#ifdef __USING_SJLJ_EXCEPTIONS__
  _Unwind_SjLj_Resume_or_Rethrow (hdr);
#else
#if defined(_LIBUNWIND_STD_ABI)
  _Unwind_RaiseException (hdr);
#else
  _Unwind_Resume_or_Rethrow (hdr);
#endif
#endif

  /* Rethrowing the exception should not return.  */
  abort();
}

/* Unwind function calls until we reach the one which used a defer
   function which called recover.  Each function which uses a defer
   statement will have an exception handler, as shown above.  */

void
__go_unwind_stack ()
{
  struct _Unwind_Exception *hdr;

  hdr = ((struct _Unwind_Exception *)
	 __go_alloc (sizeof (struct _Unwind_Exception)));
  __builtin_memcpy (&hdr->exception_class, &__go_exception_class,
		    sizeof hdr->exception_class);
  hdr->exception_cleanup = NULL;

  runtime_g ()->exception = hdr;

#ifdef __USING_SJLJ_EXCEPTIONS__
  _Unwind_SjLj_RaiseException (hdr);
#else
  _Unwind_RaiseException (hdr);
#endif

  /* Raising an exception should not return.  */
  abort ();
}

/* The rest of this code is really similar to gcc/unwind-c.c and
   libjava/exception.cc.  */

typedef struct
{
  _Unwind_Ptr Start;
  _Unwind_Ptr LPStart;
  _Unwind_Ptr ttype_base;
  const unsigned char *TType;
  const unsigned char *action_table;
  unsigned char ttype_encoding;
  unsigned char call_site_encoding;
} lsda_header_info;

static const unsigned char *
parse_lsda_header (struct _Unwind_Context *context, const unsigned char *p,
		   lsda_header_info *info)
{
  _uleb128_t tmp;
  unsigned char lpstart_encoding;

  info->Start = (context ? _Unwind_GetRegionStart (context) : 0);

  /* Find @LPStart, the base to which landing pad offsets are relative.  */
  lpstart_encoding = *p++;
  if (lpstart_encoding != DW_EH_PE_omit)
    p = read_encoded_value (context, lpstart_encoding, p, &info->LPStart);
  else
    info->LPStart = info->Start;

  /* Find @TType, the base of the handler and exception spec type data.  */
  info->ttype_encoding = *p++;
  if (info->ttype_encoding != DW_EH_PE_omit)
    {
      p = read_uleb128 (p, &tmp);
      info->TType = p + tmp;
    }
  else
    info->TType = 0;

  /* The encoding and length of the call-site table; the action table
     immediately follows.  */
  info->call_site_encoding = *p++;
  p = read_uleb128 (p, &tmp);
  info->action_table = p + tmp;

  return p;
}

/* The personality function is invoked when unwinding the stack due to
   a panic.  Its job is to find the cleanup and exception handlers to
   run.  We can't split the stack here, because we won't be able to
   unwind from that split.  */

#ifdef __ARM_EABI_UNWINDER__
/* ARM EABI personality routines must also unwind the stack.  */
#define CONTINUE_UNWINDING \
  do								\
    {								\
      if (__gnu_unwind_frame (ue_header, context) != _URC_OK)	\
	return _URC_FAILURE;					\
      return _URC_CONTINUE_UNWIND;				\
    }								\
  while (0)
#else
#define CONTINUE_UNWINDING return _URC_CONTINUE_UNWIND
#endif

#ifdef __USING_SJLJ_EXCEPTIONS__
#define PERSONALITY_FUNCTION    __gccgo_personality_sj0
#define __builtin_eh_return_data_regno(x) x
#else
#define PERSONALITY_FUNCTION    __gccgo_personality_v0
#endif

#ifdef __ARM_EABI_UNWINDER__
_Unwind_Reason_Code
PERSONALITY_FUNCTION (_Unwind_State, struct _Unwind_Exception *,
		      struct _Unwind_Context *)
  __attribute__ ((no_split_stack, flatten));

_Unwind_Reason_Code
PERSONALITY_FUNCTION (_Unwind_State state,
		      struct _Unwind_Exception * ue_header,
		      struct _Unwind_Context * context)
#else
_Unwind_Reason_Code
PERSONALITY_FUNCTION (int, _Unwind_Action, _Unwind_Exception_Class,
		      struct _Unwind_Exception *, struct _Unwind_Context *)
  __attribute__ ((no_split_stack, flatten));

_Unwind_Reason_Code
PERSONALITY_FUNCTION (int version,
		      _Unwind_Action actions,
		      _Unwind_Exception_Class exception_class,
		      struct _Unwind_Exception *ue_header,
		      struct _Unwind_Context *context)
#endif
{
  lsda_header_info info;
  const unsigned char *language_specific_data, *p, *action_record;
  _Unwind_Ptr landing_pad, ip;
  int ip_before_insn = 0;
  _Bool is_foreign;
  G *g;

#ifdef __ARM_EABI_UNWINDER__
  _Unwind_Action actions;

  switch (state & _US_ACTION_MASK)
    {
    case _US_VIRTUAL_UNWIND_FRAME:
      actions = _UA_SEARCH_PHASE;
      break;

    case _US_UNWIND_FRAME_STARTING:
      actions = _UA_CLEANUP_PHASE;
      if (!(state & _US_FORCE_UNWIND)
	  && ue_header->barrier_cache.sp == _Unwind_GetGR(context, 13))
	actions |= _UA_HANDLER_FRAME;
      break;

    case _US_UNWIND_FRAME_RESUME:
      CONTINUE_UNWINDING;
      break;

    default:
      abort();
    }
  actions |= state & _US_FORCE_UNWIND;

  is_foreign = 0;

  /* The dwarf unwinder assumes the context structure holds things like the
     function and LSDA pointers.  The ARM implementation caches these in
     the exception header (UCB).  To avoid rewriting everything we make the
     virtual IP register point at the UCB.  */
  ip = (_Unwind_Ptr) ue_header;
  _Unwind_SetGR (context, 12, ip);
#else
  if (version != 1)
    return _URC_FATAL_PHASE1_ERROR;

  is_foreign = exception_class != __go_exception_class;
#endif

  language_specific_data = (const unsigned char *)
    _Unwind_GetLanguageSpecificData (context);

  /* If no LSDA, then there are no handlers or cleanups.  */
  if (! language_specific_data)
    CONTINUE_UNWINDING;

  /* Parse the LSDA header.  */
  p = parse_lsda_header (context, language_specific_data, &info);
#ifdef HAVE_GETIPINFO
  ip = _Unwind_GetIPInfo (context, &ip_before_insn);
#else
  ip = _Unwind_GetIP (context);
#endif
  if (! ip_before_insn)
    --ip;
  landing_pad = 0;
  action_record = NULL;

#ifdef __USING_SJLJ_EXCEPTIONS__
  /* The given "IP" is an index into the call-site table, with two
     exceptions -- -1 means no-action, and 0 means terminate.  But
     since we're using uleb128 values, we've not got random access
     to the array.  */
  if ((int) ip <= 0)
    return _URC_CONTINUE_UNWIND;
  else
    {
      _uleb128_t cs_lp, cs_action;
      do
	{
	  p = read_uleb128 (p, &cs_lp);
	  p = read_uleb128 (p, &cs_action);
	}
      while (--ip);

      /* Can never have null landing pad for sjlj -- that would have
	 been indicated by a -1 call site index.  */
      landing_pad = (_Unwind_Ptr)cs_lp + 1;
      if (cs_action)
	action_record = info.action_table + cs_action - 1;
      goto found_something;
    }
#else
  /* Search the call-site table for the action associated with this IP.  */
  while (p < info.action_table)
    {
      _Unwind_Ptr cs_start, cs_len, cs_lp;
      _uleb128_t cs_action;

      /* Note that all call-site encodings are "absolute" displacements.  */
      p = read_encoded_value (0, info.call_site_encoding, p, &cs_start);
      p = read_encoded_value (0, info.call_site_encoding, p, &cs_len);
      p = read_encoded_value (0, info.call_site_encoding, p, &cs_lp);
      p = read_uleb128 (p, &cs_action);

      /* The table is sorted, so if we've passed the ip, stop.  */
      if (ip < info.Start + cs_start)
	p = info.action_table;
      else if (ip < info.Start + cs_start + cs_len)
	{
	  if (cs_lp)
	    landing_pad = info.LPStart + cs_lp;
	  if (cs_action)
	    action_record = info.action_table + cs_action - 1;
	  goto found_something;
	}
    }
#endif

  /* IP is not in table.  No associated cleanups.  */
  CONTINUE_UNWINDING;

 found_something:
  if (landing_pad == 0)
    {
      /* IP is present, but has a null landing pad.
	 No handler to be run.  */
      CONTINUE_UNWINDING;
    }

  if (actions & _UA_SEARCH_PHASE)
    {
      if (action_record == 0)
	{
	  /* This indicates a cleanup rather than an exception
	     handler.  */
	  CONTINUE_UNWINDING;
	}

      return _URC_HANDLER_FOUND;
    }

  /* It's possible for g to be NULL here for an exception thrown by a
     language other than Go.  */
  g = runtime_g ();
  if (g == NULL)
    {
      if (!is_foreign)
	abort ();
    }
  else
    {
      g->exception = ue_header;
      g->is_foreign = is_foreign;
    }

  _Unwind_SetGR (context, __builtin_eh_return_data_regno (0),
		 (_Unwind_Ptr) ue_header);
  _Unwind_SetGR (context, __builtin_eh_return_data_regno (1), 0);
  _Unwind_SetIP (context, landing_pad);
  return _URC_INSTALL_CONTEXT;
}
