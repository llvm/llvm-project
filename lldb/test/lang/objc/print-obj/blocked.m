//===-- blocked.m --------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file is for testing running "print object" in a case where another thread
// blocks the print object from making progress.  Set a breakpoint on the line in 
// my_pthread_routine as indicated.  Then switch threads to the main thread, and
// do print the lock_me object.  Since that will try to get the lock already gotten
// by my_pthread_routime thread, it will have to switch to running all threads, and
// that should then succeed.
//

#include <Foundation/Foundation.h>
#include <pthread.h>

static pthread_mutex_t test_mutex;

static void Mutex_Init (void)
{
    pthread_mutexattr_t tmp_mutex_attr;
    pthread_mutexattr_init(&tmp_mutex_attr);
    pthread_mutex_init(&test_mutex, &tmp_mutex_attr);
}

@interface LockMe :NSObject
{
  
}
- (NSString *) description;
@end

@implementation LockMe 
- (NSString *) description
{
    printf ("LockMe trying to get the lock.\n");
    pthread_mutex_lock(&test_mutex);
    printf ("LockMe got the lock.\n");
    pthread_mutex_unlock(&test_mutex);
    return @"I am pretty special.\n";
}
@end

void *
my_pthread_routine (void *data)
{
    printf ("my_pthread_routine about to enter.\n");  
    pthread_mutex_lock(&test_mutex);
    printf ("Releasing Lock.\n"); // Set a breakpoint here.
    pthread_mutex_unlock(&test_mutex);
    return NULL;
}

int
main ()
{
  pthread_attr_t tmp_attr;
  pthread_attr_init (&tmp_attr);
  pthread_t my_pthread;

  Mutex_Init ();

  LockMe *lock_me = [[LockMe alloc] init];
  pthread_create (&my_pthread, &tmp_attr, my_pthread_routine, NULL);

  pthread_join (my_pthread, NULL);

  return 0;
}
