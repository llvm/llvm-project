/*
 * ompd_test.h
 *
 *  Created on: Dec 28, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef SRC_OMPD_TEST_H_
#define SRC_OMPD_TEST_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "ompd.h"
#include "stdio.h"

/*******************************************************************************
 * NOTE: These calls are not part of OMPD. They are only used for testing.
 */

void test_print_header();
void test_CB_dmemory_alloc();
void test_CB_tsizeof_prim();

#ifdef __cplusplus
}
#endif
#endif /* SRC_OMPD_TEST_H_ */
