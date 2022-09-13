//
// Created by tanmay on 9/6/22.
//

#include "../AtomicCondition/AtomicCondition.h"

/*----------------------------------------------------------------------------*/
/* Constants                                                                  */
/*----------------------------------------------------------------------------*/
#define RESULT_DIRECTORY_NAME ".fFPTrax_results"

void fRSStoreACResult() {


  int I = 0;
  while ((uint64_t)I < StorageTable->Size) {
    if(StorageTable->FP32ACItems[I].NodeId != -1) {
      int J = 0;
      while ((uint64_t)J < FP32ResultListSize) {
        if (strcmp(StorageTable->FP32ACItems[I].XName,
                   FP32ResultList[J].XName) == 0 &&
            strcmp(StorageTable->FP32ACItems[I].YName,
                   FP32ResultList[J].YName) == 0 &&
            StorageTable->FP32ACItems[I].OP == FP32ResultList[J].OP) {
          if ((StorageTable->FP32ACItems[I].ACWRTX > FP32ResultList[J].ACWRTX &&
               StorageTable->FP32ACItems[I].ACWRTX >
                   FP32ResultList[J].ACWRTY) ||
              (StorageTable->FP32ACItems[I].ACWRTY > FP32ResultList[J].ACWRTX &&
               StorageTable->FP32ACItems[I].ACWRTY >
                   FP32ResultList[J].ACWRTY)) {
            FP32ResultList[J] = StorageTable->FP32ACItems[I];
          }
          break;
        }
        J++;
      }

      if ((uint64_t)J == FP32ResultListSize) {
        FP32ResultList[J] = StorageTable->FP32ACItems[I];
        FP32ResultListSize++;
      }
    }
    I++;
  }

  I=0;
  while ((uint64_t)I < StorageTable->Size) {
    if (StorageTable->FP64ACItems[I].NodeId != -1) {
      int J = 0;
      while ((uint64_t)J < FP64ResultListSize) {
        if (strcmp(StorageTable->FP64ACItems[I].XName,
                   FP64ResultList[J].XName) == 0 &&
            strcmp(StorageTable->FP64ACItems[I].YName,
                   FP64ResultList[J].YName) == 0 &&
            StorageTable->FP64ACItems[I].OP == FP64ResultList[J].OP) {
          if ((StorageTable->FP64ACItems[I].ACWRTX > FP64ResultList[J].ACWRTX &&
               StorageTable->FP64ACItems[I].ACWRTX >
                   FP64ResultList[J].ACWRTY) ||
              (StorageTable->FP64ACItems[I].ACWRTY > FP64ResultList[J].ACWRTX &&
               StorageTable->FP64ACItems[I].ACWRTY >
                   FP64ResultList[J].ACWRTY)) {
            FP64ResultList[J] = StorageTable->FP64ACItems[I];
          }
          break;
        }
        J++;
      }
      if ((uint64_t)J == FP64ResultListSize) {
        FP64ResultList[J] = StorageTable->FP64ACItems[I];
        FP64ResultListSize++;
      }
    }
    I++;
  }

  // Create a directory if not present
  char *DirectoryName = (char *)malloc((strlen(RESULT_DIRECTORY_NAME)+1) * sizeof(char));
  strcpy(DirectoryName, RESULT_DIRECTORY_NAME);
  fAFcreateLogDirectory(DirectoryName);

  char ExecutionId[5000];
  char FileName[5000];
  FileName[0] = '\0';
  strcpy(FileName, strcat(strcpy(FileName, DirectoryName), "/fAC_Results_"));

  fACGenerateExecutionID(ExecutionId);
  strcat(ExecutionId, ".json");

  strcat(FileName, ExecutionId);

  // Table Output
  FILE *FP = fopen(FileName, "w");
  fprintf(FP, "{\n");

  long unsigned int RecordsStored = 0;

  fprintf(FP, "\t\"FP32\": [\n");
  I = 0;

  while ((uint64_t)I < FP32ResultListSize) {
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"NodeId\":%d,\n"
                "\t\t\t\"XName\": \"%s\",\n"
                "\t\t\t\"X\": %0.7f,\n"
                "\t\t\t\"YName\": \"%s\",\n"
                "\t\t\t\"Y\": %0.7f,\n"
                "\t\t\t\"Operation\": %d,\n"
                "\t\t\t\"ACWRTX\": %0.7f,\n"
                "\t\t\t\"ACWRTY\": %0.7f,\n"
                "\t\t\t\"ACWRTXstring\": \"%s\",\n"
                "\t\t\t\"ACWRTYstring\": \"%s\"\n",
                FP32ResultList[I].NodeId,
                FP32ResultList[I].XName,
                FP32ResultList[I].X,
                FP32ResultList[I].YName,
                FP32ResultList[I].Y,
                FP32ResultList[I].OP,
                FP32ResultList[I].ACWRTX,
                FP32ResultList[I].ACWRTY,
                FP32ResultList[I].ACWRTXstring,
                FP32ResultList[I].ACWRTYstring) > 0)
      RecordsStored++;

    if (RecordsStored != FP32ResultListSize)
      fprintf(FP, "\t\t},\n");
    else
      fprintf(FP, "\t\t}\n");
    I++;
  }
  fprintf(FP, "\t],\n");

  RecordsStored = 0;

  fprintf(FP, "\t\"FP64\": [\n");
  I = 0;
  while ((uint64_t)I < FP64ResultListSize) {
    if (fprintf(FP,
                "\t\t{\n"
                "\t\t\t\"NodeId\":%d,\n"
                "\t\t\t\"XName\": \"%s\",\n"
                "\t\t\t\"X\": %0.15f,\n"
                "\t\t\t\"YName\": \"%s\",\n"
                "\t\t\t\"Y\": %0.15f,\n"
                "\t\t\t\"Operation\": %d,\n"
                "\t\t\t\"ACWRTX\": %0.15f,\n"
                "\t\t\t\"ACWRTY\": %0.15f,\n"
                "\t\t\t\"ACWRTXstring\": \"%s\",\n"
                "\t\t\t\"ACWRTYstring\": \"%s\"\n",
                FP64ResultList[I].NodeId,
                FP64ResultList[I].XName,
                FP64ResultList[I].X,
                FP64ResultList[I].YName,
                FP64ResultList[I].Y,
                FP64ResultList[I].OP,
                FP64ResultList[I].ACWRTX,
                FP64ResultList[I].ACWRTY,
                FP64ResultList[I].ACWRTXstring,
                FP64ResultList[I].ACWRTYstring) > 0)
      RecordsStored++;

    if (RecordsStored != FP64ResultListSize)
      fprintf(FP, "\t\t},\n");
    else
      fprintf(FP, "\t\t}\n");
    I++;
  }
  fprintf(FP, "\t]\n");

  fprintf(FP, "}\n");

  fclose(FP);

  printf("\nHighest AC for each operation written to file: %s\n", FileName);
}